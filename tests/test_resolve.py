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

import json
import urllib.error
from pathlib import Path

import pytest

from colab_vllm_server.resolve import ResolveError, dedupe_last_wins, parse_flags, resolve

PRESETS = sorted((Path(__file__).parents[1] / "presets").glob("*.yaml"))


def flag_values(resolved):
    return dict(parse_flags(resolved.argv[3:]))


# Flag parsing


def test_parse_flags_handles_values_switches_and_equals():
    assert parse_flags(["--a", "1", "--b", "--c=x", "--d", '{"k": 1}']) == [
        ("--a", "1"),
        ("--b", None),
        ("--c", "x"),
        ("--d", '{"k": 1}'),
    ]


def test_parse_flags_rejects_positional_arguments():
    with pytest.raises(ResolveError):
        parse_flags(["model-id"])


def test_dedupe_keeps_last_value_in_last_position():
    assert dedupe_last_wins([("--a", "1"), ("--b", None), ("--a", "2")]) == [
        ("--b", None),
        ("--a", "2"),
    ]


# Repository presets


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.stem)
def test_every_repository_preset_resolves(path):
    resolved = resolve(str(path))
    flags = flag_values(resolved)
    assert flags["--tensor-parallel-size"] == "1"
    assert flags["--host"] == "127.0.0.1"
    assert "VLLM_USE_RUST_FRONTEND" not in resolved.env


def test_bf16_a100_80g_command():
    resolved = resolve(str(Path(__file__).parents[1] / "presets" / "qwen3.8-27b-a100-80g.yaml"))
    assert resolved.argv == [
        "vllm",
        "serve",
        "Qwen/Qwen3.8-27B",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--language-model-only",
        "--kv-cache-dtype",
        "bfloat16",
        "--default-chat-template-kwargs",
        '{"enable_thinking": true, "reasoning_effort": "xhigh"}',
        "--tensor-parallel-size",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        "Qwen/Qwen3.8-27B",
    ]
    assert resolved.accelerator == "A100" and resolved.high_mem


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.stem)
def test_every_repository_preset_keeps_a_full_precision_kv_cache(path):
    assert flag_values(resolve(str(path)))["--kv-cache-dtype"] == "bfloat16"


def test_profile_borrows_a_render_as_is(preset):
    resolved = resolve(
        preset(recipe="openai/gpt-oss-120b", gpu="G4", profile="h100", frontend="python")
    )
    flags = flag_values(resolved)
    assert resolved.recipe_hw == "h100" and not resolved.exact_hw
    assert "--quantization-config.moe.activation" not in flags
    assert any("profile h100 chosen by the preset" in w for w in resolved.warnings)
    assert not any("adapted render" in w for w in resolved.warnings)


# Features, overrides, and frontend


def test_mode_uses_the_exact_profile_override(preset):
    resolved = resolve(
        preset(
            recipe="Qwen/Qwen3.8-27B",
            variant="nvfp4_nvidia",
            gpu="G4",
            features={"spec_decoding": "mtp"},
        )
    )
    assert '"num_speculative_tokens":5' in flag_values(resolved)["--speculative-config"]


def test_disabled_feature_removes_its_flags(preset):
    resolved = resolve(
        preset(recipe="Qwen/Qwen3.8-27B", gpu="A100_80G", features={"encoder_parallel": False})
    )
    assert "--mm-encoder-tp-mode" not in flag_values(resolved)


def test_overrides_win_over_the_render(preset):
    resolved = resolve(
        preset(
            recipe="Qwen/Qwen3.8-27B",
            gpu="A100_80G",
            overrides={"--max-model-len": 4096, "--enforce-eager": True},
        )
    )
    flags = flag_values(resolved)
    assert flags["--max-model-len"] == "4096" and flags["--enforce-eager"] is None


def test_python_frontend_removes_the_rust_switch(preset):
    rust = resolve(preset(name="rust", recipe="Qwen/Qwen3.8-27B", gpu="A100_80G"))
    python = resolve(
        preset(name="py", recipe="Qwen/Qwen3.8-27B", gpu="A100_80G", frontend="python")
    )
    assert rust.env.get("VLLM_USE_RUST_FRONTEND") == "1"
    assert any("Rust frontend" in w for w in rust.warnings)
    assert "VLLM_USE_RUST_FRONTEND" not in python.env


# Pre-flight checks


def test_rejects_a_variant_larger_than_the_gpu(preset):
    with pytest.raises(ResolveError, match="needs 67 GB"):
        resolve(preset(recipe="Qwen/Qwen3.8-27B", variant="default", gpu="A100"))


def test_rejects_nvfp4_before_blackwell(preset):
    with pytest.raises(ResolveError):
        resolve(preset(recipe="Qwen/Qwen3.8-27B", variant="nvfp4_nvidia", gpu="A100_80G"))


def test_vram_override_replaces_the_recipe_estimate(preset):
    with pytest.raises(ResolveError, match="needs 96 GB"):
        resolve(preset(name="est", recipe="openai/gpt-oss-120b", gpu="H100"))
    resolved = resolve(
        preset(name="real", recipe="openai/gpt-oss-120b", gpu="H100", vram_minimum_gb=72)
    )
    assert any("overridden" in w for w in resolved.warnings)


def test_rejects_a_variant_bound_to_another_generation(preset):
    with pytest.raises(ResolveError, match="only on hopper"):
        resolve(preset(recipe="zai-org/GLM-5.3", variant="w4a8", gpu="A100_80G", vram_minimum_gb=1))


def test_rejects_a_multi_node_recipe(preset):
    with pytest.raises(ResolveError, match="2 h100 nodes"):
        resolve(preset(recipe="Qwen/Qwen3-Coder-480B-A35B-Instruct", gpu="H100", vram_minimum_gb=1))


def test_rejects_unknown_names(preset):
    with pytest.raises(ResolveError, match="unknown variant"):
        resolve(preset(name="v", recipe="Qwen/Qwen3.8-27B", variant="nope", gpu="A100_80G"))
    with pytest.raises(ResolveError, match="unknown feature"):
        resolve(
            preset(name="f", recipe="Qwen/Qwen3.8-27B", gpu="A100_80G", features={"nope": True})
        )
    with pytest.raises(ResolveError, match="unknown Colab GPU"):
        resolve(preset(name="g", recipe="Qwen/Qwen3.8-27B", gpu="T4"))


# Borrowed profiles: an Ampere GPU using the Hopper render


def test_hopper_block_is_replaced_by_the_nvidia_block(preset):
    hopper = resolve(
        preset(name="h", recipe="MiniMaxAI/MiniMax-M2.5", gpu="H100", vram_minimum_gb=1)
    )
    ampere = resolve(
        preset(name="a", recipe="MiniMaxAI/MiniMax-M2.5", gpu="A100_80G", vram_minimum_gb=1)
    )
    assert hopper.env.get("VLLM_USE_DEEP_GEMM") == "0"
    assert "--attention-backend" in flag_values(hopper)
    assert "VLLM_USE_DEEP_GEMM" not in ampere.env
    assert "--attention-backend" not in flag_values(ampere)
    assert flag_values(ampere)["--compilation-config"] == '{"cudagraph_mode":"PIECEWISE"}'


def test_hopper_tuning_is_removed(preset):
    hopper = resolve(
        preset(name="h", recipe="nvidia/Llama-3.3-70B-Instruct-FP8", gpu="H100", vram_minimum_gb=1)
    )
    ampere = resolve(
        preset(
            name="a", recipe="nvidia/Llama-3.3-70B-Instruct-FP8", gpu="A100_80G", vram_minimum_gb=1
        )
    )
    assert "--async-scheduling" in flag_values(hopper)
    assert "--async-scheduling" not in flag_values(ampere)
    assert flag_values(ampere)["--kv-cache-dtype"] == "fp8"


def test_chip_specific_variant_override_is_removed(preset):
    hopper = resolve(
        preset(name="h", recipe="MiniMaxAI/MiniMax-M3-MXFP8", gpu="H100", vram_minimum_gb=1)
    )
    ampere = resolve(
        preset(name="a", recipe="MiniMaxAI/MiniMax-M3-MXFP8", gpu="A100_80G", vram_minimum_gb=1)
    )
    assert flag_values(hopper)["--max-model-len"] == "131072"
    assert "--max-model-len" not in flag_values(ampere)
    assert any("removed variant settings for h100" in w for w in ampere.warnings)


def test_recipe_without_hopper_settings_is_unchanged(preset):
    hopper = resolve(preset(name="h", recipe="Qwen/Qwen3.8-27B", gpu="H100"))
    ampere = resolve(preset(name="a", recipe="Qwen/Qwen3.8-27B", gpu="A100_80G"))
    assert hopper.argv == ampere.argv
    assert not any("adapted render" in w for w in ampere.warnings)


def test_rejects_an_accelerator_that_colab_new_does_not_accept(preset, monkeypatch):
    from colab_vllm_server import resolve as resolve_module

    gpus = resolve_module.load_gpus()
    gpus["TYPO"] = {**gpus["A100"], "accelerator": "A1000"}
    monkeypatch.setattr(resolve_module, "load_gpus", lambda: gpus)
    with pytest.raises(ResolveError, match="accepts"):
        resolve(preset(recipe="Qwen/Qwen3.8-27B", variant="int4", gpu="TYPO"))


# Preset overrides, feature removal, frontend warning, recipe errors


def test_override_values_follow_vllm_argument_syntax(preset):
    resolved = resolve(
        preset(
            recipe="Qwen/Qwen3.8-27B",
            gpu="A100_80G",
            overrides={
                "--enable-prefix-caching": False,
                "--enforce-eager": True,
                "--speculative-config": {"method": "ngram", "num_speculative_tokens": 3},
            },
        )
    )
    flags = flag_values(resolved)
    assert flags["--no-enable-prefix-caching"] is None
    assert "--enable-prefix-caching" not in flags
    assert flags["--enforce-eager"] is None
    assert json.loads(flags["--speculative-config"]) == {
        "method": "ngram",
        "num_speculative_tokens": 3,
    }


def test_disabled_feature_removes_its_hardware_specific_flags(preset):
    resolved = resolve(
        preset(
            recipe="MiniMaxAI/MiniMax-M3-MXFP8",
            gpu="H100",
            vram_minimum_gb=1,
            features={"encoder_parallel": False},
        )
    )
    flags = flag_values(resolved)
    assert "--mm-encoder-attn-backend" not in flags
    assert "--mm-processor-cache-type" not in flags


def test_rust_warning_follows_the_preset_env(preset):
    rust = resolve(
        preset(
            name="rust",
            recipe="Qwen/Qwen3.8-27B",
            gpu="A100_80G",
            frontend="python",
            env={"VLLM_USE_RUST_FRONTEND": 1},
        )
    )
    assert any("Rust frontend" in w for w in rust.warnings)


def test_unknown_recipe_is_a_resolve_error(preset, monkeypatch):
    from colab_vllm_server import recipes

    def not_found(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(recipes, "_download", not_found)
    with pytest.raises(ResolveError, match="no recipe"):
        resolve(preset(recipe="nobody/no-such-model", gpu="H100"))


def test_server_error_falls_back_to_the_cached_recipe(monkeypatch):
    from colab_vllm_server import recipes

    def unavailable(url):
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(recipes, "_download", unavailable)
    assert recipes.model("Qwen/Qwen3.8-27B", refresh=True)["hf_id"] == "Qwen/Qwen3.8-27B"


def test_quickstart_shows_the_current_command():
    quickstart = (Path(__file__).parents[1] / "docs" / "quickstart.md").read_text()
    resolved = resolve(str(Path(__file__).parents[1] / "presets" / "qwen3.8-27b-a100-80g.yaml"))
    assert resolved.command() in quickstart.splitlines()
