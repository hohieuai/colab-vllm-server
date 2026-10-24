# Get reproducible samples

This guide shows how to sample so that a rerun returns the same outputs. Read it before you run
an experiment whose results someone else must reproduce.
[Reproducibility](../explanation/reproducibility.md) explains why each step is needed.

## Set every sampling parameter

Omitted parameters take the model's defaults, not OpenAI's. For Qwen3.8-27B, the defaults are
`temperature` 1.0, `top_p` 0.95, and `top_k` 20. Set each parameter in every request:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    temperature=0.7,
    top_p=0.95,
    max_tokens=2048,
    extra_body={"top_k": 20},
)
```

## Send one request per seed

To get several samples that a rerun reproduces, send one request per seed, each with `n=1`:

```python
samples = [
    client.chat.completions.create(
        model="Qwen/Qwen3.8-27B",
        messages=messages,
        temperature=1.0,
        seed=seed,
    )
    .choices[0]
    .message.content
    for seed in range(8)
]
```

Don't combine `seed` with `n` greater than 1 when you need reproducibility. The samples differ
from each other, but a rerun of the same request can return different samples.

## Use n when reproducibility doesn't matter

If you need diverse samples and not exact reruns, ask for them in one request:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    n=8,
    temperature=1.0,
)
samples = [choice.message.content for choice in response.choices]
```

## Rerun greedy outputs on an idle server

With `temperature=0`, the output repeats only when no other requests run at the same time. To
rerun greedy outputs:

1. Stop other clients of the endpoint.
2. Send the requests one at a time.
3. Report in your method section that the greedy outputs came from sequential requests.

## Record the run

Save the report path that `serve check` prints next to your results. The report holds the
resolved command, the recipe commit, and the measured reproducibility of that deployment.

## Related pages

- [Reproducibility](../explanation/reproducibility.md)
- [Validate a deployment](validate-a-deployment.md)
- [Example: multiple samples](../../examples/python/multiple_samples.py)
