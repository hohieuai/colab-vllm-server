# Validation results

This page lists the checks that `serve check` runs, the results for the BF16 deployment, and the
load test results.

## Checks

| Check | Passes when |
|---|---|
| `auth` | A request without the API key returns 401, and the served model is listed. |
| `greedy_sanity` | Five short factual and arithmetic questions are answered correctly at temperature 0. |
| `reasoning_split` | The thinking is in `reasoning`, the answer is in `content`, no `<think>` tag leaks into the answer, and the streamed answer equals the non-streamed one. |
| `thinking_off` | `enable_thinking: false` returns no reasoning. |
| `finish_length` | A request cut off by `max_tokens` returns `finish_reason: length`. |
| `stop_sequence` | A `stop` sequence ends generation. |
| `tool_calls` | All 30 concurrent tool-call requests return `tool_calls` with the right arguments. 90% or more is a warning. |
| `tool_roundtrip` | The model uses a returned tool result in its answer. |
| `structured_json` | `json_schema` output validates against a schema that the prompt does not mention, with thinking on and off. |
| `structured_choice` | A `choice` constraint returns one of the listed strings. |
| `n_samples` | `n=4` returns 4 choices, and a repeated seeded `n=8` request returns the same samples. Distinct but not repeated samples are a warning. |
| `determinism` | Greedy output repeats on an idle server and under concurrent load, and seeded sampling repeats. A change under load only is a warning. |
| `logprobs` | Token log probabilities with 5 alternatives are returned. |
| `long_context` | A fact placed at 37% of a prompt that fills 85% of the context window is retrieved. |
| `throughput` | Always passes. Reports time to first token, single-stream speed, and speed at 16 concurrent requests. |
| `gsm8k` | 90% or more of the first 100 GSM8K test questions are correct, thinking on. Below 90% is a warning. Runs with `--accuracy`. |

## BF16 on A100 80 GB

Preset `qwen3.8-27b-a100-80g` as it is in this repository: Qwen3.8-27B in BF16, BF16 KV cache,
thinking on at `xhigh` by default, vLLM 0.32.0, Python frontend, recipe commit `49d634bd6a`. Run
from a fresh clone with the Colab CLI 0.7.5, October 2026. Result: 15 pass, 1 warn, 0 fail.

| Check | Result | Detail |
|---|---|---|
| `auth` | PASS | |
| `greedy_sanity` | PASS | 5 of 5 |
| `reasoning_split` | PASS | Streamed and non-streamed answers equal |
| `thinking_off` | PASS | |
| `finish_length` | PASS | |
| `stop_sequence` | PASS | |
| `tool_calls` | PASS | 30 of 30 |
| `tool_roundtrip` | PASS | |
| `structured_json` | PASS | 4 of 4 with thinking off, 4 of 4 with thinking on |
| `structured_choice` | PASS | |
| `n_samples` | PASS | 6 of 8 samples distinct, and the repeated seeded request returned the same samples. In the 3 earlier runs it returned different samples, so treat seeded `n` above 1 as not reproducible. |
| `determinism` | WARN | Greedy output repeated on an idle server and changed under concurrent load. Seeded `n=1` sampling repeated. |
| `logprobs` | PASS | |
| `long_context` | PASS | Fact retrieved from a 208,745-token prompt |
| `throughput` | PASS | Time to first token 0.39 s, 29 tokens per second single stream, 386 tokens per second at 16 concurrent requests |
| `gsm8k` | PASS | 97% of 100, none cut off, 438 tokens per answer |

An earlier run, before the preset set the KV cache and the reasoning effort explicitly, scored 94%
on `gsm8k` with the same other results. With 100 questions, the difference is within sampling
noise. The KV cache held 271,369 tokens in both runs, so the BF16 setting matches what vLLM chose
before.

## Frontend comparison

The same checks ran against vLLM's Rust frontend, the recipe's default, on Qwen3.8-27B in INT4
on an A100 40 GB:

| Check | Rust frontend | Python frontend |
|---|---|---|
| `n_samples` | FAIL: `Only n=1 is supported` | PASS |
| `structured_json` | FAIL: 1 of 4 answers cut off at `{\n  "` | PASS |
| `throughput` | 63 and 730 tokens per second | 63 and 730 tokens per second |
| All other checks | PASS | PASS |

## Throughput

`evals/loadtest.py` on `qwen3.8-27b-a100-80g`. Each request: a unique prompt of about 800 tokens,
exactly 300 generated tokens, thinking off, through the SSH tunnel.

| Concurrent requests | Output tokens per second | Requests per second | Time to first token, 95th percentile | Latency, median |
|---|---|---|---|---|
| 1 | 27 | 0.09 | 0.6 s | 11 s |
| 2 | 55 | 0.18 | 0.7 s | 11 s |
| 4 | 102 | 0.34 | 1.3 s | 12 s |
| 8 | 185 | 0.62 | 2.0 s | 13 s |
| 16 | 318 | 1.06 | 3.5 s | 15 s |
| 32 | 482 | 1.61 | 6.2 s | 20 s |
| 64 | 660 | 2.20 | 12 s | 29 s |
| 128 | 636 | 2.12 | 40 s | 61 s |

At 128 concurrent requests, the server ran about 69 requests with the KV cache 98% full and queued
the rest. No request failed at any level.
