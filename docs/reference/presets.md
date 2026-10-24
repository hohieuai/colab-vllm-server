# Preset reference

A preset is a YAML file in `presets/` that describes one deployment. The resolver combines it with
the recipe on [vLLM Recipes](https://recipes.vllm.ai) to build the `vllm serve` command.

## Example

```yaml
# presets/qwen3.8-27b-a100-80g.yaml
recipe: Qwen/Qwen3.8-27B
variant: default
gpu: A100_80G

features:
  text_only: true
  encoder_parallel: false

frontend: python
```

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `recipe` | string | Yes | Recipe id. The recipe page is `https://recipes.vllm.ai/<recipe>`. |
| `variant` | string | No | Key in the recipe's `variants`, such as `default` or `fp8`. Default: `default`. |
| `gpu` | string | Yes | Key in `colab_gpus.yaml`. See [GPUs](gpus.md). |
| `profile` | string | No | Recipe profile whose command to use as is, such as `h100`. For a recipe that renders no command for the GPU's own profile. Turns off the generation adaptation of [borrowed profiles](gpus.md#borrowed-profiles). |
| `features` | map | No | Recipe feature to `true`, `false`, or a mode name. `true` adds an opt-in feature. `false` removes a default-on feature. A mode name selects one mode of a feature that has modes. |
| `overrides` | map | No | vLLM flag to value. `true` adds the flag without a value. `false` adds its `--no-` form, such as `--no-enable-prefix-caching`. A map or list becomes JSON. |
| `frontend` | string | No | `python` or `rust`. Default: the recipe's default frontend. |
| `served_model_name` | string | No | The `model` value clients send. Default: the recipe id. |
| `vram_minimum_gb` | number | No | Memory the deployment needs, replacing the recipe's estimate. |
| `vllm_version` | string | No | Exact vLLM version to install. Default: the latest release. |
| `env` | map | No | Extra environment variables for the server. |

The features and variants of a recipe are listed on its page, and in
`cache/recipes/<recipe>.json` after the first `serve resolve`.

## Resolution order

The resolver builds the command in this order. A flag set by a later step replaces the value from
an earlier step.

1. The argv that recipes.vllm.ai renders for the variant's checkpoint on the GPU's recipe profile:
   `https://recipes.vllm.ai/<checkpoint>/hw/<profile>.json`. It includes the default-on
   features and the recipe's settings for that profile.
2. For a GPU with `exact: false`, the profile-specific settings are replaced with the settings for
   the GPU's own generation. See [GPUs](gpus.md#borrowed-profiles).
3. The preset's `features`.
4. The preset's `overrides`.
5. `--tensor-parallel-size 1`, `--host 127.0.0.1`, `--port 8000`, and `--served-model-name`.

The environment comes from the render, then from enabled features, then from the preset's
`frontend` and `env`.

## Pre-flight checks

`serve resolve` and `serve up` reject a preset, before any VM starts, when any of these is true:

| Check | Message starts with |
|---|---|
| The recipe id does not exist on recipes.vllm.ai. | `recipes.vllm.ai has no recipe` |
| The variant needs more memory than the GPU has. | `variant '<v>' needs <n> GB` |
| The variant's `supported_hardware` excludes the GPU's profile and architecture. | `variant '<v>' lists supported_hardware` |
| The recipe marks the profile `unsupported`. | `recipe marks <profile> unsupported` |
| The variant runs only on another GPU generation. | `variant '<v>' runs only on <generation> GPUs` |
| The precision is NVFP4 or FP4 on a GPU older than Blackwell. | `nvfp4 needs Blackwell` |
| The recipe deploys the model on several nodes. | `the recipe deploys <model> on <n> <profile> nodes` |
| recipes.vllm.ai has no render for the checkpoint and profile. | `recipes.vllm.ai has no render` |
| A variant, feature, mode, or GPU name does not exist. | `unknown variant`, `unknown feature`, `unknown mode`, or `unknown Colab GPU` |
| The GPU entry's `accelerator` is not a value that `colab new --gpu` accepts. | `colab_gpus.yaml entry '<gpu>' has accelerator` |

## Warnings

These conditions produce a warning and do not stop the preset:

| Warning | Condition |
|---|---|
| `profile <profile> chosen by the preset` | The preset sets `profile`. |
| `has no recipes profile; using the <profile> render adapted to <generation>` | The GPU has `exact: false`. |
| `adapted render: ...` | Settings were removed or applied for the GPU's generation. One warning per change. |
| `variant eligibility checked against <profile>` | The variant's `supported_hardware` names the borrowed profile. |
| `recipe does not mark <profile> as verified` | The recipe has not tested the exact profile. |
| `vram_minimum_gb overridden by preset` | The preset replaces the recipe's memory estimate. |
| `fp8 on sm80 falls back to weight-only kernels` | FP8 or MXFP4 on a GPU without native FP8 support. |
| `Rust frontend: ...` | The server will use the Rust frontend. |

## Installation

Colab VMs have no Docker. `launch.py` installs vLLM from the recipe's pip command into
`/content/colab-vllm-server/.venv`, then runs the recipe's extra install steps. When a recipe needs an
unreleased vLLM, the install uses the nightly wheel index.

After an install finishes, `launch.py` stores a hash of the install script in the environment. A
relaunch on the same VM reuses the environment only when the hash matches. If the last install
failed partway, or the script changed, for example through `vllm_version`, it deletes the
environment and installs again.
