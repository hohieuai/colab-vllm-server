# Run concurrent requests

This guide shows how to send many requests at once and how to measure what your endpoint
sustains. vLLM batches concurrent requests on the GPU, so a batch job finishes far sooner when
it keeps many requests in flight.

## Send requests concurrently

Use `AsyncOpenAI` and limit the number of requests in flight with a semaphore:

```python
import asyncio

from openai import AsyncOpenAI

client = AsyncOpenAI()
limit = asyncio.Semaphore(32)


async def ask(question: str) -> str:
    async with limit:
        response = await client.chat.completions.create(
            model="Qwen/Qwen3.8-27B",
            messages=[{"role": "user", "content": question}],
            max_tokens=512,
        )
        return response.choices[0].message.content


async def main(questions: list[str]) -> list[str]:
    return await asyncio.gather(*(ask(q) for q in questions))
```

The runnable version is [`examples/python/batch_async.py`](../../examples/python/batch_async.py).

## Pick a concurrency limit

- For batch jobs on the A100 80 GB BF16 deployment, keep 32 to 64 requests in flight. 64 gives
  the highest throughput. 32 gives about 75% of it at half the latency.
- For interactive use, keep the limit at 8 or below. The first token then arrives within about
  2 seconds.
- If your requests are longer than about 1,100 tokens, lower the limit. Each request takes more
  KV cache, so fewer fit at once.


## Measure your own workload

To find the limit for your request sizes, run the load test with your prompt and output lengths:

```bash
uv run python evals/loadtest.py qwen3.8-27b-a100-80g --input 2000 --output 1000
```

The script sends 1, 2, 4, and up to 256 concurrent requests. It stops when the 95th-percentile
time to first token exceeds 30 seconds and saves a report in `evals/results/`. Each request uses
a unique prompt, so prefix caching does not inflate the numbers.

Pick the lowest concurrency that reaches about 90% of the peak throughput. The script prints it
on its last line.

## Related pages

- [GPUs](../reference/gpus.md)
