# Copyright 2026 The colab-vllm-server Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn a preset into the exact `vllm serve` command to run on Colab.

Layering (each later step wins on a repeated flag, matching the recipes
site's last-wins dedupe):

1. argv synthesized by recipes.vllm.ai for (variant checkpoint, hardware);
   already includes default-on features and hardware overrides.
2. Preset feature toggles: opt-in features switched on (with mode and
   per-hardware args), default-on features switched off (flags removed).
3. Preset `overrides`.
4. Colab-forced flags: one GPU, loopback bind, fixed port, served name.
"""

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from colab_vllm_server import recipes

ROOT = Path(__file__).resolve().parents[2]
PORT = 8000
# Values `colab new --gpu` accepts (google-colab-cli 0.7).
COLAB_ACCELERATORS = {"T4", "L4", "G4", "H100", "A100"}

DEFAULT_PIP = "uv venv\nsource .venv/bin/activate\nuv pip install -U vllm --torch-backend auto"
NIGHTLY_PIP = (
    "uv venv\nsource .venv/bin/activate\n"
    "uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130"
)


class ResolveError(Exception):
    pass


@dataclass
class Resolved:
    preset: str
    recipe: str
    recipe_sha: str
    variant: str
    model_id: str
    precision: str
    gpu: str
    accelerator: str
    high_mem: bool
    recipe_hw: str
    exact_hw: bool
    vram_gb: int
    served_model_name: str
    argv: list
    env: dict
    install_script: str
    warnings: list = field(default_factory=list)

    def command(self) -> str:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        cmd = shlex.join(self.argv)
        return f"{env} {cmd}" if env else cmd


def load_gpus() -> dict:
    return yaml.safe_load((ROOT / "colab_gpus.yaml").read_text())["gpus"]


def load_preset(name_or_path: str) -> tuple[str, dict]:
    path = Path(name_or_path)
    if not path.exists():
        path = ROOT / "presets" / f"{name_or_path}.yaml"
    if not path.exists():
        raise ResolveError(f"preset not found: {name_or_path}")
    return path.stem, yaml.safe_load(path.read_text())


def parse_flags(tokens: list) -> list:
    """Split argv tokens into [(flag, value_or_None), ...]."""
    flags, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ResolveError(f"unexpected positional argument: {tok}")
        if "=" in tok:
            flag, value = tok.split("=", 1)
            flags.append((flag, value))
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            flags.append((tok, tokens[i + 1]))
            i += 2
        else:
            flags.append((tok, None))
            i += 1
    return flags


def dedupe_last_wins(flags: list) -> list:
    out = {}
    for flag, value in flags:
        out.pop(flag, None)
        out[flag] = value
    return list(out.items())


def flatten(flags: list) -> list:
    argv = []
    for flag, value in flags:
        argv.append(flag)
        if value is not None:
            argv.append(value)
    return argv


def _sm(arch: str) -> int:
    return int(arch.removeprefix("sm"))


def _hw_override(overrides: dict, hw: str, generation: str) -> dict:
    for key in (hw, generation, "nvidia"):
        if key in (overrides or {}):
            return overrides[key]
    return {}


def _feature_args(feature: dict, mode: str | None, variant: str, hw: str, gen: str):
    """Return (args, env) for an enabled feature, honoring modes + hw overrides."""
    if "modes" in feature:
        mode = mode or feature.get("default_mode") or next(iter(feature["modes"]))
        if mode not in feature["modes"]:
            raise ResolveError(f"unknown mode {mode!r}; choose from {list(feature['modes'])}")
        spec = feature["modes"][mode]
        if spec.get("variants") and variant not in spec["variants"]:
            raise ResolveError(f"mode {mode!r} is not available for variant {variant!r}")
        gates = spec.get("hardware") or {}
        if "unsupported" in (gates.get(hw), gates.get(gen)):
            raise ResolveError(f"mode {mode!r} is unsupported on {hw or gen}")
    else:
        spec = feature
    ho = _hw_override(spec.get("hardware_overrides"), hw, gen)
    args = ho.get("args", spec.get("args", []))
    env = ho.get("env", spec.get("env", {}))
    return list(args), dict(env)


def negate_flag(flag: str) -> str:
    """--enable-x becomes --no-enable-x, and --no-enable-x becomes --enable-x."""
    name = flag.removeprefix("--")
    return "--" + (name.removeprefix("no-") if name.startswith("no-") else f"no-{name}")


def _feature_footprint(feature: dict) -> tuple[set, set]:
    """Every flag and env var a feature can add, in any mode or hardware form."""
    specs = list(feature["modes"].values()) if "modes" in feature else [feature]
    flags, env = set(), set()
    for spec in specs:
        forms = [spec, *(spec.get("hardware_overrides") or {}).values()]
        for form in forms:
            flags |= {f for f, _ in parse_flags(list(form.get("args", [])))}
            env |= set(form.get("env") or {})
    return flags, env


def _override_value(value) -> str | None:
    """Text for a preset override value, as vLLM's argument parser expects it."""
    if value is True or value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _strip_borrowed(recipe, variant, variant_key, disabled, hw, render_gen, gen, flags, env):
    """Adapt a render made for another GPU generation to generation `gen`.

    A render for profile `hw` (generation `render_gen`) includes the overrides
    keyed by `hw` and `render_gen`, and skips the brand-wide `nvidia` block
    whenever a generation block exists. For a Colab GPU without its own
    profile, remove what was specific to the borrowed profile and apply what
    `gen` receives instead. Returns the new flags, the new env, and a list of
    the changes made.
    """
    remove, add, changes = [], [], []

    def first(overrides, keys):
        for key in keys:
            if key in overrides:
                return key, overrides[key]
        return None, None

    for scope, overrides in (
        ("recipe", recipe.get("hardware_overrides") or {}),
        ("variant", variant.get("hardware_overrides") or {}),
    ):
        if scope == "variant" and hw in overrides:
            remove.append(overrides[hw])
            changes.append(f"removed variant settings for {hw}")
        if render_gen not in overrides and gen not in overrides:
            continue  # the render used the nvidia block, which `gen` also gets
        out_key, out = first(overrides, (render_gen, "nvidia"))
        in_key, into = first(overrides, (gen, "nvidia"))
        if out_key == in_key:
            continue
        if out:
            remove.append(out)
            changes.append(f"removed {scope} settings for {out_key}")
        if into:
            add.append(into)
            changes.append(f"applied {scope} settings for {in_key}")

    opt_in = set(recipe.get("opt_in_features") or [])
    for name, feature in (recipe.get("features") or {}).items():
        if name in opt_in or name in disabled:
            continue
        rendered = _feature_args(feature, None, variant_key, hw, render_gen)
        own = _feature_args(feature, None, variant_key, "", gen)
        if rendered != own:
            remove.append({"extra_args": rendered[0], "extra_env": rendered[1]})
            add.append({"extra_args": own[0], "extra_env": own[1]})
            changes.append(f"used the {gen} form of feature {name!r}")

    dropped = set()
    for block in remove:
        names = {f for f, _ in parse_flags(list(block.get("extra_args", [])))}
        dropped |= names
        flags = [(f, v) for f, v in flags if f not in names]
        for key in block.get("extra_env", {}):
            env.pop(key, None)
    # A removed flag may also be a base setting of the recipe; restore it.
    base = parse_flags(list(recipe["model"].get("base_args", [])))
    base += parse_flags(list(variant.get("extra_args", [])))
    flags += [(f, v) for f, v in base if f in dropped]
    for block in add:
        flags += parse_flags(list(block.get("extra_args", [])))
        env.update(block.get("extra_env", {}))
    return flags, env, changes


def _install_script(recipe: dict, variant: dict, vllm_version: str | None) -> str:
    model = recipe["model"]
    pip = (model.get("install") or {}).get("pip", {})
    if pip is False:
        raise ResolveError("recipe has no pip install path (Docker only); Colab has no Docker")
    nightly = variant.get("nightly_required") or model.get("nightly_required")
    script = (pip or {}).get("command") or (NIGHTLY_PIP if nightly else DEFAULT_PIP)
    if vllm_version:
        script = script.replace("uv pip install -U vllm", f"uv pip install vllm=={vllm_version}")
    lines = [script]
    for extra in (model.get("install") or {}).get("extras", []):
        lines.append(extra["command"])
    for dep in recipe.get("dependencies", []):
        modes = dep.get("install_modes", ["pip"])
        brand = dep.get("brand")
        brands = [brand] if isinstance(brand, str) else (brand or ["NVIDIA"])
        if "pip" in modes and "NVIDIA" in brands and not dep.get("optional"):
            lines.append(dep["command"])
    return "\n".join(lines)


def resolve(preset_name: str, refresh: bool = False) -> Resolved:
    name, preset = load_preset(preset_name)
    gpus = load_gpus()
    if preset["gpu"] not in gpus:
        raise ResolveError(f"unknown Colab GPU {preset['gpu']!r}; known: {list(gpus)}")
    gpu = gpus[preset["gpu"]]
    accelerator = gpu.get("accelerator", preset["gpu"])
    if accelerator not in COLAB_ACCELERATORS:
        # `colab new` silently falls back to an A100 for an unknown --gpu value.
        raise ResolveError(
            f"colab_gpus.yaml entry {preset['gpu']!r} has accelerator {accelerator!r}; "
            f"`colab new --gpu` accepts {sorted(COLAB_ACCELERATORS)}"
        )
    hw, arch = gpu["recipe_hw"], gpu["compute_arch"]
    # `profile` borrows another profile's command as is, for a recipe that
    # renders none for this GPU. The preset author picks it on purpose, so the
    # generation adaptation for borrowed profiles does not apply.
    chosen_profile = preset.get("profile")
    if chosen_profile:
        hw = chosen_profile
    exact = gpu["exact"] and not chosen_profile
    warnings = []

    try:
        recipe = recipes.model(preset["recipe"], refresh)
    except recipes.RecipeNotFound:
        raise ResolveError(
            f"recipes.vllm.ai has no recipe {preset['recipe']!r}; "
            f"check the id at https://recipes.vllm.ai/{preset['recipe']}"
        ) from None
    variant_key = preset.get("variant", "default")
    if variant_key not in recipe["variants"]:
        raise ResolveError(
            f"unknown variant {variant_key!r}; choose from {list(recipe['variants'])}"
        )
    variant = recipe["variants"][variant_key]
    model_id = variant.get("model_id", recipe["hf_id"])
    precision = variant["precision"]

    # Fit / compatibility checks, before spending any compute units.
    if (recipe["meta"].get("hardware") or {}).get(hw) == "unsupported":
        raise ResolveError(f"recipe marks {hw} unsupported")
    supported = variant.get("supported_hardware")
    if supported and hw not in supported and f"arch:{arch}" not in supported:
        raise ResolveError(
            f"variant {variant_key!r} lists supported_hardware={supported}, not {hw}"
        )
    if supported and hw in supported and not exact:
        warnings.append(
            f"variant eligibility checked against {hw}; the recipe lists no profile "
            f"for {preset['gpu']}"
        )
    need = variant.get("vram_minimum_gb", 0)
    if "vram_minimum_gb" in preset:
        # The recipe's figure is a formula estimate (params x bytes x 1.2); a
        # preset may replace it with the measured footprint of the checkpoint.
        warnings.append(
            f"vram_minimum_gb overridden by preset: {preset['vram_minimum_gb']} GB "
            f"(recipe estimate {need} GB)"
        )
        need = preset["vram_minimum_gb"]
    if need > gpu["vram_gb"]:
        raise ResolveError(
            f"variant {variant_key!r} needs {need} GB, Colab {preset['gpu']} has {gpu['vram_gb']} GB"
        )
    sm = _sm(arch)
    if precision in ("nvfp4", "fp4") and sm < 100:
        raise ResolveError(f"{precision} needs Blackwell (sm100+), {preset['gpu']} is {arch}")
    if precision in ("fp8", "mxfp4") and sm < 89:
        warnings.append(f"{precision} on {arch} falls back to weight-only kernels (slower)")
    gen = gpu["generation"]
    bound = (variant.get("precision_hardware") or {}).get("generation")
    if bound and bound != gen:
        raise ResolveError(
            f"variant {variant_key!r} runs only on {bound} GPUs; {preset['gpu']} is {gen}"
        )
    if chosen_profile:
        warnings.append(
            f"profile {chosen_profile} chosen by the preset: using its render as is on Colab "
            f"{preset['gpu']} (not verified by the recipe for this GPU)"
        )
    elif not exact:
        warnings.append(
            f"{preset['gpu']} has no recipes profile; using the {hw} render adapted to "
            f"{gen} (not verified by the recipe)"
        )
    elif (recipe["meta"].get("hardware") or {}).get(hw) != "verified":
        warnings.append(f"recipe does not mark {hw} as verified")

    try:
        rendered = recipes.render(model_id, hw, refresh)
    except recipes.RecipeNotFound:
        raise ResolveError(
            f"recipes.vllm.ai has no render for {model_id} on {hw}; "
            f"available: {list((recipe.get('recommended_command') or {}).get('by_hardware', {}))}"
        )
    render_gen = rendered.get("hardware_profile", {}).get("generation", "")
    if rendered.get("deploy_type") == "multi_node":
        raise ResolveError(
            f"the recipe deploys {model_id} on {rendered.get('node_count')} {hw} nodes; "
            "Colab provides one GPU"
        )
    if "argv" not in rendered:
        raise ResolveError(f"recipes.vllm.ai has no single-node vllm serve command for {model_id}")
    argv = rendered["argv"]
    if argv[:2] != ["vllm", "serve"]:
        raise ResolveError(f"unexpected render argv: {argv[:3]}")
    flags = parse_flags(argv[3:])
    env = dict(rendered.get("env", {}))
    disabled = {k for k, v in (preset.get("features") or {}).items() if v is False}
    adapted = []
    if render_gen != gen and not chosen_profile:
        flags, env, adapted = _strip_borrowed(
            recipe, variant, variant_key, disabled, hw, render_gen, gen, flags, env
        )
        warnings += [f"adapted render: {c}" for c in adapted]
    own_hw = hw if exact else ""

    for feat_name, setting in (preset.get("features") or {}).items():
        feature = recipe["features"].get(feat_name)
        if feature is None:
            raise ResolveError(
                f"unknown feature {feat_name!r}; choose from {list(recipe['features'])}"
            )
        if setting is False:
            drop_flags, drop_env = _feature_footprint(feature)
            flags = [(f, v) for f, v in flags if f not in drop_flags]
            for key in drop_env:
                env.pop(key, None)
            continue
        mode = setting if isinstance(setting, str) else None
        args, fenv = _feature_args(feature, mode, variant_key, own_hw, gen)
        flags += parse_flags(args)
        env.update(fenv)

    for flag, value in (preset.get("overrides") or {}).items():
        if value is False:
            # vLLM's boolean options take a --no- form to turn them off.
            flags = [(f, v) for f, v in flags if f != flag]
            flags.append((negate_flag(flag), None))
        else:
            flags.append((flag, _override_value(value)))

    served = preset.get("served_model_name", recipe["hf_id"])
    flags += [
        ("--tensor-parallel-size", "1"),
        ("--host", "127.0.0.1"),
        ("--port", str(PORT)),
        ("--served-model-name", served),
    ]
    # Frontend pill from the recipes taxonomy: rust = VLLM_USE_RUST_FRONTEND=1,
    # python = no env. The recipe's `default_frontend` is already baked into
    # the render's env; a preset may override it.
    frontend = preset.get("frontend")
    if frontend == "python":
        env.pop("VLLM_USE_RUST_FRONTEND", None)
    elif frontend == "rust":
        env["VLLM_USE_RUST_FRONTEND"] = "1"
    elif frontend is not None:
        raise ResolveError(f"frontend must be python or rust, got {frontend!r}")
    env.update({k: str(v) for k, v in (preset.get("env") or {}).items()})
    if env.get("VLLM_USE_RUST_FRONTEND") == "1":
        warnings.append(
            "Rust frontend: n>1 unsupported and structured outputs ignored "
            "(set `frontend: python` for research use)"
        )

    return Resolved(
        preset=name,
        recipe=recipe["hf_id"],
        recipe_sha=recipe.get("source", {}).get("sha", ""),
        variant=variant_key,
        model_id=model_id,
        precision=precision,
        gpu=preset["gpu"],
        accelerator=accelerator,
        high_mem=bool(gpu.get("high_mem", False)),
        recipe_hw=hw,
        exact_hw=exact,
        vram_gb=gpu["vram_gb"],
        served_model_name=served,
        argv=["vllm", "serve", model_id] + flatten(dedupe_last_wins(flags)),
        env=env,
        install_script=_install_script(recipe, variant, preset.get("vllm_version")),
        warnings=warnings,
    )
