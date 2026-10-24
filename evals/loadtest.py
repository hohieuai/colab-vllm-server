# Copyright 2026 The colab-vllm-server Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Concurrency sweep: how much load an endpoint sustains.

  uv run python evals/loadtest.py qwen3.8-27b-a100-80g
  uv run python evals/loadtest.py qwen3.8-27b-a100-80g --levels 1,8,32,128 --input 2000 --output 500

At each concurrency level C, C workers send streaming requests back to back
for a fixed number of requests. Every request has the same shape: a unique
random prompt of ~--input tokens (unique so prefix caching cannot help) and
exactly --output generated tokens (ignore_eos), thinking off. Reported per
level: request/s, output tok/s, TTFT and end-to-end latency percentiles,
per-token decode time (TPOT), and errors.

vLLM queues requests beyond what fits (max_num_seqs / KV cache) rather than
rejecting them, so saturation shows up as flat throughput with rising TTFT.
All traffic goes through the SSH tunnel, which is part of what is measured.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "river",
    "mountain",
    "cloud",
    "signal",
    "vector",
    "matrix",
    "orange",
    "window",
    "garden",
    "silver",
    "rocket",
    "ocean",
    "planet",
    "engine",
    "forest",
    "stone",
]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def make_prompt(n_tokens, rng):
    # ~1.3 tokens per word for this vocabulary; exact length is reported back
    # by usage.prompt_tokens.
    words = [rng.choice(WORDS) for _ in range(int(n_tokens / 1.3))]
    return f"[{rng.random():.12f}] " + " ".join(words) + "\n\nContinue this sequence."


async def one_request(client, model, prompt, out_tokens):
    t0 = time.perf_counter()
    first = None
    n_out = 0
    n_in = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=out_tokens,
        temperature=1.0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={
            "ignore_eos": True,
            "min_tokens": out_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content and first is None:
            first = time.perf_counter()
        if chunk.usage:
            n_out = chunk.usage.completion_tokens
            n_in = chunk.usage.prompt_tokens
    end = time.perf_counter()
    first = first or end
    return {
        "ttft": first - t0,
        "e2e": end - t0,
        "out": n_out,
        "in": n_in,
        "tpot": (end - first) / max(1, n_out - 1),
    }


async def run_level(client, model, conc, n_requests, in_tokens, out_tokens, seed):
    rng = random.Random(seed)
    prompts = [make_prompt(in_tokens, rng) for _ in range(n_requests)]
    queue = asyncio.Queue()
    for p in prompts:
        queue.put_nowait(p)
    results, errors = [], []

    async def worker():
        while True:
            try:
                p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(await one_request(client, model, p, out_tokens))
            except Exception as e:  # noqa: BLE001 - count every failed request
                errors.append(f"{type(e).__name__}: {str(e)[:120]}")

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(conc)))
    wall = time.perf_counter() - t0
    if not results:
        return {"concurrency": conc, "errors": len(errors), "error_examples": errors[:3]}
    return {
        "concurrency": conc,
        "requests": len(results),
        "errors": len(errors),
        "error_examples": errors[:3],
        "wall_s": round(wall, 1),
        "req_per_s": round(len(results) / wall, 2),
        "out_tok_per_s": round(sum(r["out"] for r in results) / wall),
        "mean_prompt_tokens": round(statistics.mean(r["in"] for r in results)),
        "ttft_p50": round(pct([r["ttft"] for r in results], 50), 2),
        "ttft_p95": round(pct([r["ttft"] for r in results], 95), 2),
        "e2e_p50": round(pct([r["e2e"] for r in results], 50), 1),
        "e2e_p95": round(pct([r["e2e"] for r in results], 95), 1),
        "tpot_ms_p50": round(1000 * pct([r["tpot"] for r in results], 50), 1),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("--levels", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--input", type=int, default=1000, help="approx prompt tokens")
    ap.add_argument("--output", type=int, default=300, help="generated tokens per request")
    ap.add_argument("--requests-per-worker", type=int, default=3)
    ap.add_argument("--min-requests", type=int, default=16)
    ap.add_argument(
        "--stop-ttft",
        type=float,
        default=30.0,
        help="stop the sweep once p95 TTFT exceeds this many seconds",
    )
    args = ap.parse_args()

    state = json.loads((ROOT / "runs" / f"{args.preset}.state.json").read_text())
    resolved = json.loads((Path(state["run_dir"]) / "resolved.json").read_text())
    client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{state['port']}/v1",
        api_key=state["api_key"],
        timeout=1800,
        max_retries=0,
    )
    model = state["served_model_name"]
    print(
        f"[load] {resolved['model_id']} on Colab {resolved['gpu']}; "
        f"~{args.input} in / {args.output} out tokens, thinking off"
    )
    header = (
        f"{'conc':>5} {'req/s':>6} {'out tok/s':>9} {'TTFT p50':>9} {'TTFT p95':>9} "
        f"{'e2e p50':>8} {'e2e p95':>8} {'TPOT ms':>8} {'err':>4}"
    )
    print(header)
    rows = []
    for i, conc in enumerate(int(x) for x in args.levels.split(",")):
        n = max(args.min_requests, conc * args.requests_per_worker)
        row = await run_level(client, model, conc, n, args.input, args.output, seed=i)
        rows.append(row)
        if "req_per_s" not in row:
            print(f"{conc:>5}  all requests failed: {row['error_examples']}")
            break
        print(
            f"{conc:>5} {row['req_per_s']:>6} {row['out_tok_per_s']:>9} {row['ttft_p50']:>8}s "
            f"{row['ttft_p95']:>8}s {row['e2e_p50']:>7}s {row['e2e_p95']:>7}s "
            f"{row['tpot_ms_p50']:>8} {row['errors']:>4}",
            flush=True,
        )
        if row["ttft_p95"] > args.stop_ttft:
            print(f"[load] p95 TTFT above {args.stop_ttft}s: stopping the sweep")
            break

    measured = [r for r in rows if "out_tok_per_s" in r]
    out = ROOT / "evals" / "results" / f"load-{args.preset}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"resolved": resolved, "args": vars(args), "levels": rows}, indent=1))
    if not measured:
        print(f"[load] every request failed; report: {out}")
        return
    best = max(measured, key=lambda r: r["out_tok_per_s"])
    knee = next(
        (
            r
            for r in rows
            if "out_tok_per_s" in r and r["out_tok_per_s"] >= 0.9 * best["out_tok_per_s"]
        ),
        best,
    )
    print(
        f"[load] peak {best['out_tok_per_s']} out tok/s at concurrency {best['concurrency']}; "
        f"90% of peak already at concurrency {knee['concurrency']} "
        f"(TTFT p95 {knee['ttft_p95']}s, TPOT {knee['tpot_ms_p50']} ms)"
    )
    print(f"[load] report: {out}")


if __name__ == "__main__":
    asyncio.run(main())
