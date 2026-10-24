# API reference

The endpoint is vLLM's OpenAI-compatible server. Requests and responses use the OpenAI shapes.
This page lists the supported endpoints and parameters, the vLLM extensions, and the differences
from the OpenAI API. Every behavior on this page was checked against the `qwen3.8-27b-a100-80g`
deployment (vLLM 0.32.0, Python frontend).

## Connection

| Item | Value |
|---|---|
| Base URL | `http://127.0.0.1:<port>/v1`. `serve up` and `serve env` print it. |
| Authentication | `Authorization: Bearer <key>`. The key is `vllm` unless `VLLM_API_KEY` is set. |
| Model name | The preset's served model name, for example `Qwen/Qwen3.8-27B`. `GET /v1/models` lists it. |
| Context window | `max_model_len` in `GET /v1/models`. 262,144 tokens for `qwen3.8-27b-a100-80g`. |

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /v1/models` | Supported |
| `POST /v1/chat/completions` | Supported |
| `POST /v1/completions` | Supported |
| `POST /tokenize`, `POST /detokenize` | vLLM extension. Converts between text or messages and token ids. |
| `POST /v1/responses` | Not tested |
| Embeddings, images, audio, files, and batches | Not available for chat models |

## Message roles

| Role | Description |
|---|---|
| `system` | Instructions for the whole conversation. Must be the first message. |
| `developer` | Treated as `system`. |
| `user` | Input from the user. |
| `assistant` | An earlier model reply. |
| `tool` | A function result, with the `tool_call_id` it answers. |

A `system` message in any position other than first returns HTTP 400.

## Chat Completions parameters

| Parameter | Status |
|---|---|
| `messages` | Supported |
| `max_tokens`, `max_completion_tokens` | Supported |
| `temperature`, `top_p` | Supported. Defaults come from the model. See [Sampling defaults](#sampling-defaults). |
| `stop` | Supported |
| `seed` | Supported |
| `n` | Supported with `frontend: python` |
| `stream`, `stream_options.include_usage` | Supported |
| `tools`, `tool_choice` | Supported |
| `response_format` with `json_schema` or `json_object` | Supported with `frontend: python` |
| `logprobs`, `top_logprobs` | Supported. `top_logprobs` above 20 returns HTTP 400. |
| `presence_penalty`, `frequency_penalty`, `logit_bias` | Supported |
| `parallel_tool_calls`, `user` | Accepted |

## Sampling defaults

Omitted sampling parameters take the values in the model's `generation_config.json`, not the
OpenAI defaults. For Qwen3.8-27B:

| Parameter | Default |
|---|---|
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| `top_k` | 20 |

## Tool choice

| `tool_choice` | Behavior |
|---|---|
| `"auto"` | The model decides whether to call a function. Default when `tools` is set. |
| `"required"` | The model calls at least one function. |
| `"none"` | The model calls no function. |
| `{"type": "function", "function": {"name": "<name>"}}` | The model calls the named function. |

Tool calls are parsed with vLLM's `qwen3_coder` parser for Qwen3.8. Tool call ids start with
`chatcmpl-tool-`.

## Finish reasons

| `finish_reason` | Meaning |
|---|---|
| `stop` | The model ended its answer, or a `stop` sequence matched. |
| `length` | Generation reached `max_tokens` or the context window. The output is cut off. |
| `tool_calls` | The model called one or more functions. |

## vLLM extensions

Send these as top-level JSON fields. In the OpenAI SDKs, pass them in `extra_body`.

| Field | Description |
|---|---|
| `chat_template_kwargs` | Arguments for the chat template. See [Chat template arguments](#chat-template-arguments). |
| `top_k`, `min_p`, `repetition_penalty` | More sampling controls. |
| `min_tokens`, `ignore_eos` | Set a minimum output length, or generate until `max_tokens`. |
| `structured_outputs` | Constraints beyond JSON Schema: `choice`, `regex`, `grammar`, or `json`. |
| `prompt_logprobs` | Log probabilities of the prompt tokens. Completions API only. |
| `skip_special_tokens` | Set to `false` to keep special tokens in the output. |

## Chat template arguments

For Qwen3.8, `chat_template_kwargs` accepts:

| Argument | Values | Default |
|---|---|---|
| `enable_thinking` | `true` or `false` | `true` |
| `reasoning_effort` | `low`, `medium`, or `xhigh` | `xhigh` |

With thinking on, the template adds a system prompt that states the reasoning effort.

## Response fields that differ from OpenAI

| Field | Description |
|---|---|
| `choices[].message.reasoning` | The model's thinking. `content` holds only the answer. Not part of the OpenAI schema. |
| `choices[].delta.reasoning` | The thinking in a streamed chunk. |
| `usage.completion_tokens_details.reasoning_tokens` | Number of thinking tokens. |

## Errors

Errors use the OpenAI error shape, so the SDKs raise their usual exceptions:

```json
{"error": {"message": "...", "type": "BadRequestError", "code": 400}}
```

| Status | Cause | SDK exception |
|---|---|---|
| 400 | Invalid JSON or parameters, `max_tokens` above the context window, or a misplaced `system` message | `BadRequestError` |
| 401 | Missing or wrong API key. The body is `{"error": "Unauthorized"}`, not the OpenAI shape. | `AuthenticationError` |
| 404 | Unknown model name. The message lists the valid names. | `NotFoundError` |
