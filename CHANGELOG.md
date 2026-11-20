# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.0.2] - 2026-11-21

Supports the `colab` CLI 0.7.6 or later from PyPI, including 0.8.0. `serve up` checks the CLI's
version and the tunnel's SSH key before it starts a VM, and an unreadable keyring is reported as
such. Also adds a logo.

### Added

- A logo: a paragraph of streamed text drawn as dots, the newest tokens largest, in black and
  white. It sits at the top of the README, and `scripts/make_logo.py` writes it.
- `COLAB_SSH_KEY` names the SSH key for the tunnel, for a key not called `~/.ssh/id_ed25519` or
  `~/.ssh/id_ecdsa`. `serve` passes it to both `colab ssh` and `ssh`.

### Changed

- `serve up` checks that the colab CLI is version 0.7.6 or later before it touches Colab, and
  stops with the command that upgrades it. With 0.7.5 and older, the launch's `--timeout 0` was a
  deadline that had already passed, and the VM started anyway.
- `serve up` and `serve connect` check for the tunnel's SSH key pair before they touch Colab.
  Before, a missing key showed only after the VM had started, when the tunnel failed five times
  with `no SSH public key found` and the VM kept running.
- Install the `colab` CLI from PyPI with `uv tool install google-colab-cli`. Version 0.7.6 is the
  first release with what colab-vllm-server needs: `--timeout 0` means no time limit, and
  `colab ssh -- CMD` runs a single command. The README, the quickstart, the dev container and the
  "not installed" message no longer install it from source.
- The README, the quickstart and the agent skill clone this release, `v0.0.2`, instead of `main`.
- Troubleshooting covers the CLI's newer HTTP 429 message, `SSH session limit reached`.

### Fixed

- A locked OS keyring is no longer reported as a missing sign-in. Since 0.7.6 the colab CLI keeps
  its sign-in in the keyring, and when it can't read it, it starts a new sign-in. `serve` now says
  that the keyring could not be read, quotes the CLI's reason, and says to unlock it.

## [0.0.1] - 2026-11-13

First release. Builds installed from `main` before this release reported version 0.1.0. The
Changed list describes what differs from those builds.

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
- A dev container, VS Code settings, and issue and pull request templates.

### Changed

- `serve up` passes `--timeout 0` to `colab exec`, matching google-colab-cli idle-timeout semantics
  (no limit while the server runs). Needs a CLI build with that behavior (after 0.7.5 / `#185`).
- Docs and the operator skill tell you to run shell commands with `colab ssh -s NAME -- CMD` after
  freeing the tunnel, instead of `colab console`, and describe the SSH session limit as
  plan-dependent.
- CI runs one job, on Linux with Python 3.11, for pull requests and on demand. It cancels
  superseded runs, times out after 10 minutes, and caches uv. Dependabot version updates are off.
  `.github/CODEOWNERS` routes review to the maintainer.

[Unreleased]: https://github.com/hohieuai/colab-vllm-server/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/hohieuai/colab-vllm-server/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/hohieuai/colab-vllm-server/releases/tag/v0.0.1
