# Get structured outputs

This guide shows how to make the model's answer match a JSON Schema. The server enforces the
schema while it decodes, so the output always parses and always has the required keys.

Structured outputs need a preset with `frontend: python`. Every preset in this repository sets
it. [How colab-vllm-server works](../explanation/how-it-works.md#why-presets-use-the-python-frontend)
explains why.

## Parse the answer into a Pydantic model

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class CalendarEvent(BaseModel):
    name: str
    day: str
    participants: list[str]


completion = client.beta.chat.completions.parse(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {
            "role": "user",
            "content": "Extract the event: Alice and Bob meet for lunch at Noma on Friday.",
        }
    ],
    response_format=CalendarEvent,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
event = completion.choices[0].message.parsed
```

The SDK turns the Pydantic model into a JSON Schema and parses the answer. The runnable version
is [`examples/python/structured_outputs.py`](../../examples/python/structured_outputs.py).

## Pass a JSON Schema in the request

Without the SDK helper, put the schema in `response_format`:

```json
"response_format": {
  "type": "json_schema",
  "json_schema": {
    "name": "calendar_event",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "day": {"type": "string"},
        "participants": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["name", "day", "participants"],
      "additionalProperties": false
    }
  }
}
```

The full request is [`examples/curl/chat_completions_structured_outputs.sh`](../../examples/curl/chat_completions_structured_outputs.sh).

If you need valid JSON but have no fixed schema, use `"response_format": {"type": "json_object"}`.

## Restrict the answer to fixed labels

For classification and judge labels, list the allowed answers with the vLLM `structured_outputs`
extension:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Classify the sentiment: I loved the film."}],
    extra_body={"structured_outputs": {"choice": ["positive", "negative", "neutral"]}},
)
```

The answer is always one of the listed strings.

## Use a schema with thinking on

With thinking on, the schema applies to the answer and not to the reasoning. You can keep
thinking on for structured outputs.

## Related pages

- [Call functions](call-functions.md)
- [API reference: vLLM extensions](../reference/api.md#vllm-extensions)
