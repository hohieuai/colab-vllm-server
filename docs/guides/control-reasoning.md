# Control reasoning

Qwen3.8 thinks before it answers. The server returns the thinking in `message.reasoning` and the
answer in `message.content`. This guide shows how to read both, set how long the model thinks,
and turn thinking off.

## Read the reasoning and the answer

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {
            "role": "user",
            "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
            "than the ball. How much does the ball cost?",
        }
    ],
    max_tokens=8192,
)

message = response.choices[0].message
print("Reasoning:", getattr(message, "reasoning", None))
print("Answer:", message.content)
```

`reasoning` is a vLLM field, so the Python SDK stores it as an extra attribute. Read it with
`getattr`. In a streamed response, read `delta.reasoning`.

## Set the reasoning effort

To make the model think less, pass `reasoning_effort` to the chat template:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
```

The values are `low`, `medium`, and `xhigh`. Without the argument, the template uses `xhigh`. If
you report results, record the effort level, because the template adds it to the system prompt.

## Turn thinking off

For short factual tasks, classification, and large batch jobs, turn thinking off:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

The response then has no reasoning, and the model answers directly.

## Give thinking enough tokens

Thinking counts toward `max_tokens`. If the budget runs out during thinking, `content` is empty
and `finish_reason` is `length`.

1. Set `max_tokens` to 8,192 or more for requests with thinking on.
2. Check `finish_reason` before you use `content`.
3. If answers keep running out, lower `reasoning_effort` or raise `max_tokens`.

## Related pages

- [API reference: chat template arguments](../reference/api.md#chat-template-arguments)
- [Call functions](call-functions.md)
