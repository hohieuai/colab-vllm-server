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

"""Conformance + quality checks for a running endpoint.

Each check probes one property a research pipeline silently depends on
(reasoning split, tool-call parsing, schema enforcement, n>1 sampling,
determinism, long context, accuracy). Results go to the terminal and to
<run_dir>/check-<time>.json so a reported number can be traced back to the
exact server configuration it came from.

Statuses: PASS, WARN (works but with a caveat worth knowing), FAIL.
"""

import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from openai import OpenAI

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
THINK_MAX_TOKENS = 16384

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class Checker:
    def __init__(self, state: dict):
        self.state = state
        self.base = f"http://127.0.0.1:{state['port']}"
        self.key = state["api_key"]
        self.model = state["served_model_name"]
        self.client = OpenAI(base_url=self.base + "/v1", api_key=self.key, timeout=1800)
        self.results = []

    # -- helpers -------------------------------------------------------------

    def chat(self, content, **kw):
        messages = content if isinstance(content, list) else [{"role": "user", "content": content}]
        return self.client.chat.completions.create(model=self.model, messages=messages, **kw)

    def raw_post(self, path, body, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), headers=headers
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)

    def record(self, name, status, detail, **data):
        self.results.append({"check": name, "status": status, "detail": detail, **data})
        print(f"  {status:<4}  {name:<24} {detail}", flush=True)

    def run(self, name, fn):
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a crashing check is a failing check
            self.record(name, "FAIL", f"{type(e).__name__}: {str(e)[:200]}")
        self.results[-1]["seconds"] = round(time.time() - t0, 1)

    # -- checks --------------------------------------------------------------

    def check_auth(self):
        try:
            req = urllib.request.Request(self.base + "/v1/models")
            urllib.request.urlopen(req, timeout=10)
            unauth = 200
        except urllib.error.HTTPError as e:
            unauth = e.code
        models = self.client.models.list().data
        ids = [m.id for m in models]
        ok = unauth == 401 and self.model in ids
        self.record(
            "auth", "PASS" if ok else "FAIL", f"no-key -> HTTP {unauth}; served models {ids}"
        )

    def check_greedy_sanity(self):
        cases = [
            ("What is 17 * 23? Reply with the number only.", "391"),
            ("What is the capital of Australia? One word.", "Canberra"),
            ("What is the chemical symbol for gold? Reply with the symbol only.", "Au"),
            ("Sort these numbers ascending, comma-separated: 42, 7, 19, 3", "3, 7, 19, 42"),
            ("Thủ đô của Việt Nam là thành phố nào? Trả lời ngắn gọn.", "Hà Nội"),
        ]
        bad = []
        for q, want in cases:
            out = self.chat(q, temperature=0, max_tokens=50, extra_body=NO_THINK)
            text = out.choices[0].message.content or ""
            if want.lower().replace(" ", "") not in text.lower().replace(" ", ""):
                bad.append((q, text[:80]))
        self.record(
            "greedy_sanity",
            "PASS" if not bad else "FAIL",
            f"{len(cases) - len(bad)}/{len(cases)} correct",
            failures=bad,
        )

    def check_reasoning_split(self):
        q = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball? Answer in one short sentence."
        r = self.chat(q, temperature=0, max_tokens=THINK_MAX_TOKENS)
        m = r.choices[0].message
        reasoning = getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None) or ""
        content = m.content or ""
        leaked = "<think>" in content or "</think>" in content
        stream_content, stream_reasoning = [], []
        for ch in self.chat(q, temperature=0, max_tokens=THINK_MAX_TOKENS, stream=True):
            if ch.choices:
                d = ch.choices[0].delta
                stream_content.append(d.content or "")
                stream_reasoning.append(getattr(d, "reasoning", None) or "")
        same = "".join(stream_content).strip() == content.strip()
        answer_ok = any(a in content.lower() for a in ("0.05", "5 cents", "five cents"))
        ok = bool(reasoning) and bool(content) and not leaked and answer_ok
        status = "PASS" if ok and same else ("WARN" if ok else "FAIL")
        self.record(
            "reasoning_split",
            status,
            f"reasoning {len(reasoning)} chars, answer {content.strip()[:60]!r}, "
            f"think-tag leak={leaked}, stream==non-stream={same}",
            finish=r.choices[0].finish_reason,
        )

    def check_thinking_off(self):
        r = self.chat(
            "Name three primary colors.", temperature=0, max_tokens=200, extra_body=NO_THINK
        )
        m = r.choices[0].message
        reasoning = getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None) or ""
        content = m.content or ""
        ok = not reasoning and "<think>" not in content and content.strip()
        self.record(
            "thinking_off",
            "PASS" if ok else "FAIL",
            f"reasoning {len(reasoning)} chars, content {content.strip()[:60]!r}",
        )

    def check_finish_length(self):
        r = self.chat("Count from 1 to 500, comma-separated.", max_tokens=16, extra_body=NO_THINK)
        fr = r.choices[0].finish_reason
        self.record(
            "finish_length",
            "PASS" if fr == "length" else "FAIL",
            f"max_tokens=16 -> finish_reason={fr!r}",
        )

    def check_stop_sequence(self):
        r = self.chat(
            "Count from 1 to 20, one number per line, nothing else.",
            temperature=0,
            max_tokens=100,
            stop=["7"],
            extra_body=NO_THINK,
        )
        text = r.choices[0].message.content or ""
        ok = "7" not in text and "6" in text
        self.record(
            "stop_sequence",
            "PASS" if ok else "FAIL",
            f"stop=['7'] -> {text.strip()[-20:]!r}, finish={r.choices[0].finish_reason}",
        )

    def check_tool_calls(self, n=30):
        cities = ["Hanoi", "Melbourne", "Tokyo", "Paris", "Hue", "Nairobi"]

        def one(i):
            city = cities[i % len(cities)]
            r = self.chat(
                f"What's the weather like in {city} right now?",
                tools=[WEATHER_TOOL],
                max_tokens=THINK_MAX_TOKENS,
                seed=i,
            )
            calls = r.choices[0].message.tool_calls or []
            if not calls or calls[0].function.name != "get_weather":
                return ("no_call", r.choices[0].message.content)
            try:
                args = json.loads(calls[0].function.arguments)
            except json.JSONDecodeError:
                return ("bad_json", calls[0].function.arguments)
            return (
                ("ok", None)
                if args.get("city", "").lower() == city.lower()
                else ("wrong_arg", args)
            )

        with ThreadPoolExecutor(10) as ex:
            outcomes = list(ex.map(one, range(n)))
        ok = sum(o[0] == "ok" for o in outcomes)
        failures = [o for o in outcomes if o[0] != "ok"][:5]
        rate = ok / n
        status = "PASS" if rate == 1 else ("WARN" if rate >= 0.9 else "FAIL")
        self.record(
            "tool_calls",
            status,
            f"{ok}/{n} parsed with correct arguments",
            failures=[(k, str(v)[:200]) for k, v in failures],
        )

    def check_tool_roundtrip(self):
        msgs = [{"role": "user", "content": "What's the weather in Hanoi?"}]
        r = self.chat(msgs, tools=[WEATHER_TOOL], max_tokens=THINK_MAX_TOKENS, temperature=0)
        call = r.choices[0].message.tool_calls[0]
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [call.model_dump()]},
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(
                    {"city": "Hanoi", "temp_c": 31, "condition": "thunderstorms"}
                ),
            },
        ]
        r2 = self.chat(msgs, tools=[WEATHER_TOOL], max_tokens=THINK_MAX_TOKENS, temperature=0)
        text = r2.choices[0].message.content or ""
        ok = "31" in text and ("thunder" in text.lower() or "storm" in text.lower())
        self.record(
            "tool_roundtrip",
            "PASS" if ok else "FAIL",
            f"answer uses tool result: {text.strip()[:80]!r}",
        )

    def check_structured_json(self):
        schema = {
            "type": "object",
            "properties": {
                "zz_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "zz_tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["alpha", "beta"]},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": ["zz_score", "zz_tags"],
            "additionalProperties": False,
        }

        def valid(text):
            try:
                d = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                return False
            return (
                set(d) == {"zz_score", "zz_tags"}
                and isinstance(d["zz_score"], int)
                and 1 <= d["zz_score"] <= 5
                and len(d["zz_tags"]) == 3
                and all(t in ("alpha", "beta") for t in d["zz_tags"])
            )

        # Prompts deliberately do not mention JSON: only the grammar can make
        # the output conform.
        prompts = [
            "Write a haiku about rain.",
            "Explain photosynthesis briefly.",
            "Tell me a joke.",
            "What is 2+2?",
        ]
        rf = {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": schema, "strict": True},
        }
        results = {}
        for mode, extra, max_tokens in (
            ("thinking_off", NO_THINK, 300),
            ("thinking_on", {}, THINK_MAX_TOKENS),
        ):
            outs = [
                self.chat(
                    p, response_format=rf, temperature=0, max_tokens=max_tokens, extra_body=extra
                )
                .choices[0]
                .message.content
                for p in prompts
            ]
            results[mode] = (
                sum(valid(o) for o in outs),
                [str(o)[:80] for o in outs if not valid(o)],
            )
        ok = all(v[0] == len(prompts) for v in results.values())
        self.record(
            "structured_json",
            "PASS" if ok else "FAIL",
            ", ".join(f"{k} {v[0]}/{len(prompts)} valid" for k, v in results.items()),
            failures={k: v[1][:3] for k, v in results.items()},
        )

    def check_structured_choice(self):
        r = self.chat(
            "Is the Pacific the largest ocean?",
            temperature=0,
            max_tokens=10,
            extra_body={**NO_THINK, "structured_outputs": {"choice": ["AFFIRM_Q", "DENY_Q"]}},
        )
        text = (r.choices[0].message.content or "").strip()
        self.record(
            "structured_choice",
            "PASS" if text in ("AFFIRM_Q", "DENY_Q") else "FAIL",
            f"output {text!r}",
        )

    def check_n_samples(self):
        r = self.chat(
            "Give me a random animal name.",
            n=4,
            temperature=1.0,
            max_tokens=20,
            extra_body=NO_THINK,
        )
        outs = [c.message.content for c in r.choices]
        # Seeded n>1 requests: samples should differ within a request, and an
        # identical repeat should reproduce them. The second property did not
        # hold reliably on vLLM 0.32, so it is reported as a WARN.
        q = "Write one random 12-digit number. Reply with digits only."

        def seeded():
            r = self.chat(q, n=8, temperature=1.0, seed=7, max_tokens=20, extra_body=NO_THINK)
            return [c.message.content for c in r.choices]

        first, second = seeded(), seeded()
        distinct = len(set(first))
        reproducible = first == second
        status = "FAIL" if len(outs) != 4 or distinct == 1 else ("PASS" if reproducible else "WARN")
        self.record(
            "n_samples",
            status,
            f"n=4 returned {len(outs)} choices; seeded n=8: {distinct}/8 distinct, "
            f"repeat reproduces samples={reproducible}"
            + ("" if reproducible else " (use one n=1 request per seed for reproducibility)"),
        )

    def check_determinism(self):
        q = "List the first 15 prime numbers, then explain in three sentences why there are infinitely many primes."

        def greedy():
            return (
                self.chat(q, temperature=0, max_tokens=300, extra_body=NO_THINK)
                .choices[0]
                .message.content
            )

        alone = [greedy() for _ in range(3)]

        def noise(i):
            self.chat(f"Write a long story about city #{i}.", max_tokens=400, extra_body=NO_THINK)

        with ThreadPoolExecutor(12) as ex:
            bg = [ex.submit(noise, i) for i in range(10)]
            loaded = [ex.submit(greedy) for _ in range(2)]
            loaded = [f.result() for f in loaded]
            for f in bg:
                f.result()
        seeded = [
            self.chat(
                "Invent a name for a new planet.",
                temperature=0.9,
                seed=1234,
                max_tokens=30,
                extra_body=NO_THINK,
            )
            .choices[0]
            .message.content
            for _ in range(2)
        ]
        same_alone = len(set(alone)) == 1
        same_loaded = all(x == alone[0] for x in loaded)
        same_seed = seeded[0] == seeded[1]
        status = (
            "PASS"
            if same_alone and same_loaded and same_seed
            else ("WARN" if same_alone and same_seed else "FAIL")
        )
        self.record(
            "determinism",
            status,
            f"greedy repeat={same_alone}, greedy under batch load={same_loaded}, "
            f"seeded sampling repeat={same_seed}",
        )

    def check_logprobs(self):
        r = self.chat(
            "Say hello.", max_tokens=5, logprobs=True, top_logprobs=5, extra_body=NO_THINK
        )
        lp = r.choices[0].logprobs
        ok = bool(lp and lp.content and len(lp.content[0].top_logprobs) == 5)
        self.record(
            "logprobs",
            "PASS" if ok else "FAIL",
            f"token logprobs with top-5 alternatives present={ok}",
        )

    def check_long_context(self):
        max_len = self.client.models.list().data[0].model_extra.get("max_model_len", 32768)
        target = int(max_len * 0.85)
        filler = "The grass is green. The sky is blue. The sun is bright. "
        needle = "The secret passcode for the vault is 7-4-1-9-ZEBRA. "
        probe = self.raw_post("/tokenize", {"model": self.model, "prompt": filler})
        per = len(probe["tokens"])
        n_fill = max(1, (target - 200) // per)
        chunks = [filler] * n_fill
        chunks.insert(int(n_fill * 0.37), needle)
        prompt = (
            "".join(chunks)
            + "\n\nWhat is the secret passcode for the vault? Reply with the passcode only."
        )
        r = self.chat(prompt, temperature=0, max_tokens=30, extra_body=NO_THINK)
        text = r.choices[0].message.content or ""
        ok = "7-4-1-9-ZEBRA" in text.replace(" ", "")
        self.record(
            "long_context",
            "PASS" if ok else "FAIL",
            f"needle at 37% of {r.usage.prompt_tokens} tokens -> {text.strip()[:40]!r}",
        )

    def check_throughput(self):
        t0 = time.time()
        first = None
        n = 0
        for ch in self.chat(
            "Write a 400-word essay about rivers.",
            max_tokens=600,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=NO_THINK,
        ):
            if ch.choices and ch.choices[0].delta.content and first is None:
                first = time.time()
            if ch.usage:
                n = ch.usage.completion_tokens
        single = n / (time.time() - first)

        def one(i):
            return self.chat(
                f"Write a 300-word story about robot #{i}.", max_tokens=500, extra_body=NO_THINK
            ).usage.completion_tokens

        t1 = time.time()
        with ThreadPoolExecutor(16) as ex:
            total = sum(ex.map(one, range(16)))
        agg = total / (time.time() - t1)
        self.record(
            "throughput",
            "PASS",
            f"TTFT {first - t0:.2f}s, single stream {single:.0f} tok/s, "
            f"16 concurrent {agg:.0f} tok/s",
            ttft_s=round(first - t0, 2),
            single_tok_s=round(single),
            agg16_tok_s=round(agg),
        )

    def check_gsm8k(self, n=100):
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k"
            f"&config=main&split=test&offset=0&length={n}"
        )
        with urllib.request.urlopen(url, timeout=60) as resp:
            rows = [r["row"] for r in json.load(resp)["rows"]]

        def gold(row):
            return row["answer"].split("####")[-1].strip().replace(",", "")

        def one(row):
            r = self.chat(
                row["question"] + "\n\nPut the final numeric answer after '####'.",
                max_tokens=THINK_MAX_TOKENS,
            )
            text = r.choices[0].message.content or ""
            m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
            pred = m[-1].replace(",", "") if m else None
            try:
                hit = pred is not None and float(pred) == float(gold(row))
            except ValueError:
                hit = False
            return hit, r.choices[0].finish_reason, r.usage.completion_tokens

        with ThreadPoolExecutor(16) as ex:
            out = list(ex.map(one, rows))
        acc = sum(h for h, _, _ in out) / len(out)
        truncated = sum(f == "length" for _, f, _ in out)
        mean_tok = sum(t for _, _, t in out) / len(out)
        self.record(
            "gsm8k",
            "PASS" if acc >= 0.9 else "WARN",
            f"{acc:.1%} on first {len(out)} test questions (thinking on), "
            f"{truncated} truncated, {mean_tok:.0f} tokens/answer",
            accuracy=acc,
            n=len(out),
            truncated=truncated,
        )


CHECKS = [
    "auth",
    "greedy_sanity",
    "reasoning_split",
    "thinking_off",
    "finish_length",
    "stop_sequence",
    "tool_calls",
    "tool_roundtrip",
    "structured_json",
    "structured_choice",
    "n_samples",
    "determinism",
    "logprobs",
    "long_context",
    "throughput",
]


def run_checks(state: dict, only=None, accuracy=False) -> Path:
    known = [*CHECKS, "gsm8k"]
    unknown = [n for n in only or [] if n not in known]
    if unknown:
        raise SystemExit(f"[check] unknown check {unknown}; choose from {known}")
    checker = Checker(state)
    names = only or CHECKS + (["gsm8k"] if accuracy else [])
    resolved = json.loads((Path(state["run_dir"]) / "resolved.json").read_text())
    print(
        f"[check] {state['preset']}: {resolved['model_id']} on Colab {resolved['gpu']}, "
        f"frontend={'rust' if resolved['env'].get('VLLM_USE_RUST_FRONTEND') == '1' else 'python'}"
    )
    for name in names:
        checker.run(name, getattr(checker, f"check_{name}"))
    counts = {s: sum(r["status"] == s for r in checker.results) for s in ("PASS", "WARN", "FAIL")}
    print(f"[check] {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
    report = Path(state["run_dir"]) / f"check-{datetime.now():%Y%m%d-%H%M%S}.json"
    report.write_text(json.dumps({"resolved": resolved, "results": checker.results}, indent=1))
    print(f"[check] report: {report}")
    return report
