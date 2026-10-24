# GPU reference

`colab_gpus.yaml` lists the Colab GPUs that presets can name. Each entry maps a Colab GPU to a
hardware profile from the [vLLM Recipes taxonomy](https://github.com/vllm-project/recipes/blob/main/taxonomy.yaml).

## GPUs

| `gpu` | Card | Memory | Generation | Recipe profile | Exact |
|---|---|---|---|---|---|
| `G4` | RTX PRO 6000 Blackwell | 96 GB | Blackwell, sm120 | `rtx_pro_6000` | Yes |
| `H100` | H100 | 80 GB | Hopper, sm90 | `h100` | Yes |
| `A100_80G` | A100-SXM4-80GB | 80 GB | Ampere, sm80 | `h100` | No |
| `A100` | A100-SXM4-40GB | 40 GB | Ampere, sm80 | `h100` | No |
| `L4` | L4 | 24 GB | Ada, sm89 | `h100` | No |

## A100 machine shapes

Colab selects the A100 memory size by machine shape:

| Request | Card | System memory |
|---|---|---|
| `colab new --gpu A100` | A100-SXM4-40GB | Standard |
| `colab new --gpu A100 --high-mem` | A100-SXM4-80GB | 167 GB |

`A100_80G` sets `high_mem: true`. Measured October 2026.

## Recipe coverage

A preset resolves only when recipes.vllm.ai renders a command for the GPU's profile. Of the 217
recipe pages on recipes.vllm.ai in October 2026, with the default variant:

| `gpu` | Resolve | Main reason for rejection |
|---|---|---|
| `H100` | 78 | Model larger than one 80 GB GPU |
| `A100_80G` | 78 | Model larger than one 80 GB GPU |
| `G4` | 6 | `rtx_pro_6000` is a restricted profile. A recipe renders it only when the recipe declares it. |

For a recipe without a command for the GPU, a preset can name another profile with `profile`, and
the resolver uses that command as is. `gpt-oss-120b-g4` uses the H100 command on G4 this way.

## Borrowed profiles

The taxonomy has no Ampere or L4 profile, and only one recipe renders the Ada profile `l40s`.
A100, A100_80G, and L4 therefore borrow the `h100` render with `exact: false`.

A render for `h100` contains settings that the recipe keys to the `hopper` generation or to the
`h100` chip. When `generation` differs from the profile's generation, the resolver:

1. Removes the recipe's `hopper` settings and the variant's `hopper` and `h100` settings.
2. Applies the settings for the GPU's own generation. If the recipe has none, it applies the
   brand-wide `nvidia` settings that the `hopper` block had replaced.
3. Uses the GPU's own generation for default-on features whose arguments depend on hardware.
4. Restores any base setting of the recipe that step 1 removed.
5. Rejects a variant whose `precision_hardware` names another generation, and a feature mode that
   is `unsupported` on the GPU's generation.

The recipe maintainers have not tested borrowed profiles. Validate a borrowed deployment with
`serve check`.

## Fields

| Field | Description |
|---|---|
| `accelerator` | Value for `colab new --gpu`: `T4`, `L4`, `G4`, `H100`, or `A100`. Default: the entry's key. The resolver rejects any other value, because `colab new` silently gives an A100 for an unknown one. |
| `high_mem` | `true` to pass `--high-mem`. |
| `recipe_hw` | Recipe profile whose render is used. |
| `exact` | `true` when the profile is the same chip as the Colab GPU. |
| `generation` | GPU generation: `blackwell`, `hopper`, `ada`, or `ampere`. |
| `vram_gb` | Memory per GPU in GB. The launch script checks it with `nvidia-smi`. |
| `compute_arch` | CUDA compute capability, such as `sm80`. NVFP4 needs `sm100` or newer. Native FP8 needs `sm89` or newer. |
