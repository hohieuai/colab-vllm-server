---
allowed-tools: Bash(uv run serve resolve:*), Bash(curl -sL https://recipes.vllm.ai/*), Write(presets/*), Read
description: Add a preset for a model from recipes.vllm.ai
argument-hint: <recipe id or recipes.vllm.ai URL> <Colab GPU>
---

## Context

- Colab GPUs: @colab_gpus.yaml
- An existing preset: @presets/qwen3.8-27b-a100-80g.yaml
- Preset fields: @docs/reference/presets.md

## Your task

Add a preset for: $ARGUMENTS

1. Fetch `https://recipes.vllm.ai/<recipe id>.json` and list its variants with
   `vram_minimum_gb` and `supported_hardware`.
2. Pick the variant that fits the GPU's `vram_gb`. If none fits, stop and say which GPU would fit.
3. Write `presets/<model>-<gpu>.yaml` with `recipe`, `variant`, `gpu`, and `frontend: python`. Add a
   short comment that says why this variant fits.
4. Run `uv run serve resolve <name>` and report the command and every warning.
5. If the resolver rejects the preset, explain the reason and the fix. Set `vram_minimum_gb` only
   when you have the checkpoint's real size, and cite it in a comment.

Don't run `serve up`. Starting a VM costs compute units.
