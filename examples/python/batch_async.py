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

"""Many requests concurrently with AsyncOpenAI.

The server batches concurrent requests; around 32-64 in flight gives the best
throughput on the A100 80GB BF16 deployment (see evals/loadtest.py).
"""

import asyncio

from openai import AsyncOpenAI

client = AsyncOpenAI()
MAX_IN_FLIGHT = 32

questions = [
    f"What is {a} * {b}? Reply with the number only." for a in range(11, 21) for b in (7, 13)
]


async def ask(question: str, limit: asyncio.Semaphore) -> str:
    async with limit:
        response = await client.chat.completions.create(
            model="Qwen/Qwen3.8-27B",
            messages=[{"role": "user", "content": question}],
            temperature=0,
            max_tokens=32,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content.strip()


async def main():
    limit = asyncio.Semaphore(MAX_IN_FLIGHT)
    answers = await asyncio.gather(*(ask(q, limit) for q in questions))
    for q, a in zip(questions, answers):
        print(f"{q:<45} {a}")


asyncio.run(main())
