# Add a deployment

This guide shows how to serve another model, or the same model on another GPU, by adding a
preset. The command itself comes from the model's recipe on [vLLM Recipes](https://recipes.vllm.ai),
so a preset holds only what you choose.

## Choose the model and the variant

1. Open the model's page on [recipes.vllm.ai](https://recipes.vllm.ai). The page address is the
   recipe id, for example `https://recipes.vllm.ai/openai/gpt-oss-120b` for `openai/gpt-oss-120b`.
2. Pick a variant whose memory need fits one Colab GPU. Colab gives one GPU per VM, so a recipe
   that needs several GPUs or nodes can't run. [GPUs](../reference/gpus.md) lists the memory of
   each Colab GPU.
3. If the model is gated on Hugging Face, accept its license there and put your token in
   `.env.local` as `HF_TOKEN`.

## Write the preset

Create `presets/<name>.yaml`. Name it after the model and the GPU, such as
`llama-3.1-8b-instruct-a100.yaml`:

```yaml
recipe: meta-llama/Llama-3.1-8B-Instruct
variant: default
gpu: A100
frontend: python
```

Always set `frontend: python`. [How colab-vllm-server works](../explanation/how-it-works.md#why-presets-use-the-python-frontend)
explains why. [Presets](../reference/presets.md) lists every field.

## Resolve it

Print the command. This runs on your computer and costs nothing:

```bash
uv run serve resolve llama-3.1-8b-instruct-a100
```

If the resolver rejects the preset, the message names the reason:

- If the variant needs more memory than the GPU has, pick a smaller variant or a larger GPU.
- If the recipe's memory estimate is above the checkpoint's real size, set `vram_minimum_gb` in
  the preset to the measured footprint. Add a comment that says where the number comes from.
- If the recipe deploys the model on several nodes, the model can't run on Colab.
- If the variant runs only on another GPU generation, pick another variant or GPU.

Read the warnings too. A warning that the render was adapted means the GPU borrows another
GPU's profile. Validate such a deployment before you rely on it.

## Start and validate it

```bash
uv run serve up llama-3.1-8b-instruct-a100
uv run serve check llama-3.1-8b-instruct-a100
```

## Add a Colab GPU

If Colab offers a GPU that `colab_gpus.yaml` does not list, add an entry:

```yaml
gpus:
  MY_GPU:
    accelerator: A100   # value for `colab new --gpu`
    high_mem: false     # pass --high-mem
    recipe_hw: h100     # profile from the recipes taxonomy
    exact: false        # true only if the profile is the same chip
    generation: ampere
    vram_gb: 40
    compute_arch: sm80
```

Set `recipe_hw` to the same chip if the [recipes taxonomy](https://github.com/vllm-project/recipes/blob/main/taxonomy.yaml)
has it. Otherwise, use `h100` and set `exact: false`. [GPUs](../reference/gpus.md#fields) describes
each field.

## Related pages

- [Presets](../reference/presets.md)
- [Validate a deployment](validate-a-deployment.md)
