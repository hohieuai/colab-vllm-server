# colab-vllm-server plugin for Claude Code

This plugin teaches Claude Code to operate [colab-vllm-server](https://github.com/hohieuai/colab-vllm-server),
which runs an OpenAI-compatible vLLM endpoint on a Google Colab GPU. With it, Claude Code can start
an endpoint, send it a test message, validate it before an experiment, reconnect the tunnel, and
stop the VM when the work is done, so that no VM keeps using compute units by mistake.

## Install

```
/plugin install colab-vllm-server --marketplace hohieuai/colab-vllm-server
```

## What it adds

| Component | Description |
|---|---|
| `colab-vllm-server` skill | When to use each `serve` command, the cost rules, and the Colab pitfalls: one SSH connection per VM, no `colab exec` while the server runs, and single quotes for messages in zsh. |

## Requirements

The skill runs the `serve` command from a clone of the colab-vllm-server repository. The computer
needs uv, the Colab CLI 0.7.5 or later signed in with `colab sessions`, and Colab compute units.
The [README](https://github.com/hohieuai/colab-vllm-server#get-started) lists the setup steps.
