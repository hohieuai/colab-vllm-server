# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `serve resolve` builds the `vllm serve` command for a preset from the rendering that
  recipes.vllm.ai publishes for the checkpoint and hardware profile, and rejects presets that
  can't run on the chosen Colab GPU before a VM starts.
- `serve up` creates the Colab VM, installs vLLM, starts the server, and opens an SSH tunnel to a
  local port. `--restart` relaunches on the same VM and keeps the installed environment.
- `serve env` prints `OPENAI_BASE_URL` and `OPENAI_API_KEY`. `--write` sets them in `.env.local`.
- `serve check` tests a running endpoint: authentication, reasoning parsing, tool calls,
  structured outputs, `n` samples, determinism, log probabilities, long context, throughput, and
  GSM8K accuracy. Reports are saved with the run.
- `serve chat` sends one message to a running endpoint and streams the answer. `--think` shows
  the model's reasoning.
- `serve connect` reopens the SSH tunnel to a running endpoint, for example after the computer
  slept.
- `serve status` reports whether the API answers, whether the tunnel runs, and whether the server
  log is still streaming.
- `serve down` stops the endpoint and releases the VM.
- `serve up` checks that the colab CLI is signed in before it starts, and explains a failed
  `colab new`. The resolver rejects an `accelerator` that `colab new --gpu` does not accept.
- Presets for Qwen3.8-27B in BF16 on an A100 80 GB, gpt-oss-120b on a G4 and on an H100, and
  Llama 3.1 8B Instruct on an A100 40 GB, all configured for the highest output quality: BF16 KV
  cache and the highest server-side reasoning effort.
- The `profile` preset field uses another recipe profile's command as is, for a recipe without a
  command for the GPU.
- GPUs without their own recipe profile, such as the A100 and the L4, use the H100 render with the
  Hopper-specific and H100-specific settings removed and the settings for their own generation
  applied.
- `evals/loadtest.py` measures throughput and latency across concurrency levels.
- curl and Python examples for each API feature, and documentation.
- A Claude Code plugin with a skill for operating endpoints, published through
  `.claude-plugin/marketplace.json`, and `CLAUDE.md` with project commands for contributors.
- A dev container, VS Code settings, issue and pull request templates, Dependabot, and a workflow
  that runs Claude Code on `@claude` mentions.
