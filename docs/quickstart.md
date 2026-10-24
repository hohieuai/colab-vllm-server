# Quickstart

In this tutorial, we start Qwen3.8-27B on a Colab A100 80 GB, send it a request with curl and
with the OpenAI Python SDK, and stop it. It takes about 20 minutes. Most of that time is the first
download of the model.

## Before you start

You need:

- A computer with macOS or Linux. The Colab CLI does not support Windows.
- A Google account with Colab compute units and access to the A100. The A100 80 GB is the
  High-RAM machine shape, which needs Colab Pro or Pro+. It uses about 7 compute units per hour
  while it runs.
- `git` and OpenSSH, which macOS and most Linux distributions include.

You don't need to install Python. uv installs the version each tool needs.

## Install the tools

First, install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal so that `uv` is on your `PATH`, and check it:

```bash
uv --version
```

Next, install the Colab CLI. colab-vllm-server needs version 0.7.5 or later:

```bash
uv tool install google-colab-cli
colab version
```

The second command prints `Version: 0.7.5` or a later version.

Now get colab-vllm-server and go into its folder. Run every later command from this folder:

```bash
git clone https://github.com/hohieuai/colab-vllm-server
cd colab-vllm-server
```

## Sign in to Colab

Sign in to Colab. The command prints a URL. Open it, approve access, and paste the code back into the
terminal:

```bash
colab sessions
```

When the sign-in works, the command lists your Colab sessions. With none running, it prints:

```
[colab] No active sessions found on server.
```

## Check your SSH key

The tunnel signs in to the VM with your SSH key. List the public key:

```bash
ls ~/.ssh/id_ed25519.pub
```

If the file does not exist, create a key:

```bash
ssh-keygen -t ed25519
```

## Preview the server command

Each deployment is a preset in [`presets/`](../presets). Print the command that the preset for
this tutorial resolves to. This runs on your computer and costs nothing. The first `uv run` also
installs colab-vllm-server's dependencies, which takes a few seconds:

```bash
uv run serve resolve qwen3.8-27b-a100-80g
```

The output ends with the `vllm serve` command that runs on the VM:

```
vllm serve Qwen/Qwen3.8-27B --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only --kv-cache-dtype bfloat16 --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' --tensor-parallel-size 1 --host 127.0.0.1 --port 8000 --served-model-name Qwen/Qwen3.8-27B
```

## Start the endpoint

Start the deployment:

```bash
uv run serve up qwen3.8-27b-a100-80g
```

The command creates the VM, installs vLLM, downloads about 55 GB of weights, starts the server,
and opens the tunnel. This takes about 15 minutes. To follow the progress, open a second terminal
and run:

```bash
tail -f ~/colab-runs/vllm-qwen3-8-27b-a100-80g.log
```

When the endpoint is ready, the first terminal prints:

```
[serve] ready: Qwen/Qwen3.8-27B on Colab A100_80G
  export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
  export OPENAI_API_KEY=vllm
```

## Point your shell at the endpoint

Load the two variables into your shell:

```bash
eval "$(uv run serve env qwen3.8-27b-a100-80g)"
echo $OPENAI_BASE_URL
```

You should see `http://127.0.0.1:8000/v1`.

## Send a first message

Send a message with `serve chat`. It prints the answer as the model writes it:

```bash
uv run serve chat qwen3.8-27b-a100-80g 'Hello!'
```

The model replies, in different words each time. The last line reports the token count, the
time, and `finish_reason=stop`.

## Send a request with curl

Send a chat request:

```bash
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 1024,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The server answers with JSON. The answer is in `choices[0].message.content`:

```
"content": "Hello! How can I help you today?"
```

The `chat_template_kwargs` field turns off the model's thinking, so the answer comes back faster.

## Send a request with Python

Save this as `first_request.py`:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

Run it:

```bash
uv run --with openai python first_request.py
```

The script prints the model's reply to `Hello!`.

## Stop the endpoint

The VM uses compute units until you stop it. Stop it now:

```bash
uv run serve down qwen3.8-27b-a100-80g
```

The command prints:

```
[serve] qwen3.8-27b-a100-80g stopped and VM released
```

## Next steps

You have started an endpoint, called it two ways, and released the VM. From here:

- [Generate text](guides/generate-text.md) covers conversations and streaming.
