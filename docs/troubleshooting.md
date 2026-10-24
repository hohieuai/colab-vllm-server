# Troubleshooting

## Fix a JSON decode error from curl

If the server answers `JSON decode error` with `Invalid control character`, the request body has
a line break inside a string. This happens when a long command is copied from a narrow window
and the copy wraps the line.

Copy the command from the file instead, or run the example script:

```bash
bash examples/curl/chat_completions.sh
```

## Fix "command not found: #" in zsh

If zsh prints `command not found: #` or `number expected` after you paste commands, the paste
included comment lines. Interactive zsh does not treat `#` as a comment by default. Paste only
the commands, or turn comments on:

```bash
setopt interactivecomments
```

## Fix an Unauthorized error

If a request returns `{"error": "Unauthorized"}`, the API key is missing or wrong.

1. Load the variables for the endpoint: `eval "$(uv run serve env <preset>)"`.
2. Check that `echo $OPENAI_API_KEY` prints the key, which is `vllm` unless you set
   `VLLM_API_KEY`.

An endpoint started before the default key changed to `vllm` keeps its old random key until it
restarts. `serve env` always prints the key the running endpoint uses.

## Fix a connection error

If a request fails with `Connection refused` or `Connection reset`, nothing answers on the local
port. This often happens after the computer slept, because sleep drops the SSH tunnel.

1. Run `uv run serve status`.
2. If `api` is `unreachable`, reopen the tunnel: `uv run serve connect <preset>`.
3. If `serve connect` reports that the server does not answer, read the end of
   `~/colab-runs/<session>.log`, and run `colab sessions` to check that the VM still exists.

During startup, `api` is `unreachable` until vLLM has loaded the model. The log shows the
progress.

## Understand log=stopped

If `serve status` shows `log=stopped` while `api=ready`, the server runs and only the copy of its
output to the local log has ended. The local `colab exec` that copies it stops when its connection
to the VM drops. The server keeps running on the VM. To get a live log again, restart the endpoint
with `uv run serve up <preset> --restart`.

## Fix "Already-active SSH session (HTTP 429)"

A Colab runtime accepts one `colab ssh` connection at a time, and the endpoint's tunnel uses it.
While the tunnel is open, `colab ssh` to the same session fails with HTTP 429.

To run a shell command on the VM, use `colab console`. It opens a shell through the VM's terminal,
not through SSH or the notebook kernel, so it works while the endpoint runs:

```bash
echo "nvidia-smi --query-gpu=name,memory.used --format=csv" | colab console -s vllm-qwen3-8-27b-a100-80g
```

The output contains terminal control bytes. Filter it with `grep -a` for the line you need.

Don't use `colab exec` for this. The endpoint's server runs inside a notebook cell that never
finishes, so the kernel stays busy and a new `colab exec` never starts. In testing, it ignored
`--timeout 15` and did not return. If you started one, stop it with Ctrl+C.

## Fix "the colab CLI is not signed in"

Run `colab sessions` in a terminal. It prints a URL. Open it, approve access, and paste the code
back. Then run the `serve` command again.

## Fix a failed colab new

If `serve up` reports that `colab new` failed, Colab refused the VM. An HTTP 400 means the account
has no quota or entitlement for that GPU or machine shape. Run `colab usage` to check your compute
units, and pick a GPU your Colab plan includes.

## Fix "Colab assigned a smaller GPU"

If the server log ends with `expected ~80 GB for A100_80G, got 40 GiB`, Colab gave the standard
A100 instead of the 80 GB card. Stop the VM and start it again:

```bash
uv run serve down qwen3.8-27b-a100-80g
uv run serve up qwen3.8-27b-a100-80g
```

The `A100_80G` GPU requests the High-RAM shape, which is the 80 GB card. Availability varies by
Colab plan.

## Fix an empty answer with finish_reason "length"

If `content` is empty and `finish_reason` is `length`, the model spent `max_tokens` on thinking.
Raise `max_tokens`, lower the reasoning effort, or turn thinking off. See
[Control reasoning](guides/control-reasoning.md#give-thinking-enough-tokens).

## Fix "Only n=1 is supported"

The server runs vLLM's Rust frontend. Add `frontend: python` to the preset and restart the
endpoint with `uv run serve up <preset> --restart`.

## Understand a rejected preset

`serve resolve` names the reason in one line, for example:

```
[serve] cannot resolve: variant 'default' needs 67 GB, Colab A100 has 40 GB
```

[Add a deployment](guides/add-a-deployment.md#resolve-it) lists what to do for each reason.

## Stop paying for a VM you no longer use

A running VM uses compute units until it stops. List your endpoints and stop the ones you don't
need:

```bash
uv run serve status
uv run serve down <preset>
```

To see every Colab VM on your account, including ones started outside colab-vllm-server, run
`colab sessions`.
