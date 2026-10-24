---
name: colab-vllm-server
description: Run an OpenAI-compatible vLLM endpoint on a Google Colab GPU with colab-vllm-server. Use when asked to start, test, call, check, reconnect, or stop a model endpoint on Colab, to add a preset for a model from recipes.vllm.ai, or to send requests to such an endpoint. Works alongside the colab-operator and colab-monitor skills, and takes precedence over them for VMs that run an endpoint.
---

# Operate colab-vllm-server

colab-vllm-server starts vLLM on a Colab VM and connects it to this computer through an SSH
tunnel. The `serve` command does everything. A running endpoint costs Colab compute units every
hour until it is stopped, so always stop what you start.

## Before any command

- Run every `serve` command from the repository root, with `uv run serve ...`. Outside the
  repository, `uv run` fails with `Failed to spawn: serve`. If the repository is missing, clone it:
  `git clone https://github.com/hohieuai/colab-vllm-server`.
- Check that the Colab CLI is signed in with `colab sessions`. If it asks for an authorization code,
  stop and ask the user to run `colab sessions` in their own terminal. Never paste codes for them.
- Tell the user the cost before `serve up`: the A100 80 GB uses about 7 compute units per hour, and
  the first start takes 15 to 20 minutes.

## Commands

| Task | Command |
|---|---|
| Show the server command, offline and free | `uv run serve resolve <preset>` |
| Start the endpoint | `uv run serve up <preset>` |
| Send a test message | `uv run serve chat <preset> 'Hello!'` |
| Print the client variables | `uv run serve env <preset>` |
| Check whether it answers | `uv run serve status` |
| Reopen the tunnel after sleep | `uv run serve connect <preset>` |
| Validate before an experiment | `uv run serve check <preset>` |
| Stop it and release the VM | `uv run serve down <preset>` |

Presets are the files in `presets/`, all configured for the highest output quality. The one
validated end to end is `qwen3.8-27b-a100-80g`. For gpt-oss, pass `reasoning_effort="high"` in
requests, because vLLM has no server-side default for it.

`serve up` blocks until the server answers. Run it in the background and wait for the line
`[serve] ready`. The server log is `~/colab-runs/<session>.log`.

## Rules

- **Stop what you start.** When the task is done, run `uv run serve down <preset>`, then confirm
  with `colab sessions` that no VM is left. Ask before leaving an endpoint running.
- **Quote messages with single quotes.** In zsh, a double-quoted message ending in `!` leaves the
  quote open.
- **Don't use `colab exec` on a running endpoint's VM.** The server occupies the notebook kernel, so
  `colab exec` never returns. To run a shell command on the VM, pipe it to
  `colab console -s <session>`.
- **Don't open `colab ssh` to a running endpoint's VM.** The tunnel holds the only SSH slot, and a
  second connection fails with HTTP 429.
- **Report sampling settings.** The server's defaults come from the model, not from OpenAI. When
  results matter, pass `temperature`, `top_p`, and `max_tokens` explicitly.

## Works with colab-operator and colab-monitor

This skill works alongside the Colab CLI's `colab-operator` skill and the `colab-monitor` skill.
For a VM that runs an endpoint, the rules here take precedence:

- **Stop endpoints with `serve down`, not `colab stop`.** `colab-operator` and `colab-monitor` say
  to finish with `colab stop -s NAME`. For an endpoint, `serve down` also stops the local tunnel and
  log stream, removes the state file, and confirms in `colab sessions` that the VM is gone.
- **Don't use `colab exec` on an endpoint's VM.** `colab-operator` prefers `colab exec`, but the
  server holds the kernel, so the command never returns. Use `colab console`.
- **Don't pass `--config` to `colab` for endpoint sessions.** `serve` uses the CLI's default
  session file, which is also what `colab-monitor` reads.

The server log is already where `colab-monitor` reads logs: `~/colab-runs/<session>.log`, named
after the session, for example `vllm-qwen3-8-27b-a100-80g`. Endpoint VMs appear in its dashboard
with no extra setup.

## Call the endpoint from code

Load the variables, then use any OpenAI SDK:

```bash
eval "$(uv run serve env <preset>)"
```

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

## Add a preset

1. Find the model on https://recipes.vllm.ai and pick a variant that fits one Colab GPU.
2. Write `presets/<model>-<gpu>.yaml` with `recipe`, `variant`, `gpu`, and `frontend: python`.
3. Run `uv run serve resolve <name>` and fix any rejection it reports.

The full documentation is in `docs/`. Start with `docs/reference/command-line.md` and
`docs/troubleshooting.md`.
