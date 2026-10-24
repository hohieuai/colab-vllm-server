# Generate text

This guide shows how to send chat requests, continue a conversation, stream a response, and
complete raw text. It assumes a running endpoint and the two variables from
`eval "$(uv run serve env <preset>)"`.

## Send a chat request

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    temperature=0.7,
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

The same request with curl is in [`examples/curl/chat_completions.sh`](../../examples/curl/chat_completions.sh).
To find the value for `model`, call `GET /v1/models` with
[`examples/curl/models.sh`](../../examples/curl/models.sh).

## Continue a conversation

The API keeps no state between requests. To continue a conversation, send the whole history:

```python
messages = [{"role": "user", "content": "Name a sorting algorithm."}]
reply = client.chat.completions.create(model="Qwen/Qwen3.8-27B", messages=messages)

messages.append(reply.choices[0].message)
messages.append({"role": "user", "content": "What is its worst-case complexity?"})
reply = client.chat.completions.create(model="Qwen/Qwen3.8-27B", messages=messages)
```

Put the `system` message first. The server rejects a `system` message in any other position.

## Stream the response

To print tokens as the model produces them, set `stream=True`:

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Write a four-line poem about rain."}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

With `include_usage`, the last chunk holds the token counts and has an empty `choices` list. The
full example is [`examples/python/streaming.py`](../../examples/python/streaming.py).

## Complete raw text

To send a prompt exactly as written, without the chat template, use the Completions API. Use it
for few-shot prompts written for a base model or for log-likelihood scoring:

```python
response = client.completions.create(
    model="Qwen/Qwen3.8-27B",
    prompt="The three primary colors are",
    max_tokens=32,
    temperature=0,
)
print(response.choices[0].text)
```

## Detect a truncated answer

Before you use an answer, check `finish_reason`. If it is `length`, the output stopped at
`max_tokens` or at the context window. Raise `max_tokens` and send the request again. The
[API reference](../reference/api.md#finish-reasons) lists every value.

## Related pages

- [Control reasoning](control-reasoning.md)
- [API reference](../reference/api.md)
