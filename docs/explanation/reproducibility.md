# Reproducibility

An experiment is reproducible when the same request returns the same output on a rerun. With
vLLM, that holds for some kinds of request and not for others. This page explains which, why,
and what to report.

## What was measured

On the `qwen3.8-27b-a100-80g` deployment with vLLM 0.32.0:

| Request | Same output on a rerun |
|---|---|
| `n=1`, `seed` set, `temperature` above 0 | Yes |
| `n` above 1, `seed` set | Not reliably. The samples differ from each other, but a rerun returned the same samples in only 2 of 7 trials. |
| `temperature=0`, no other requests running | Yes |
| `temperature=0`, other requests running at the same time | No |

`serve check` measures these again on each deployment, in the `n_samples` and `determinism`
checks.

## Why greedy output changes under load

vLLM serves many requests at once by putting them in one batch on the GPU. The batch size changes
from moment to moment, and the GPU kernels that vLLM picks depend on that size. Two kernels can
add the same numbers in a different order. Floating-point addition is not associative, so the
results differ in the last bits.

Usually the difference doesn't matter. But greedy decoding takes the single most likely token,
and when two tokens are almost tied, a last-bit difference can pick the other one. From that token
on, the output follows a different path. The same request alone on the server always meets the
same batch shape, so it repeats.

## Why seeded n samples don't repeat

The measurements show the effect, not its cause. The likely cause is the same as for greedy
output. With `n` above 1, the samples of one request share the batch with every other request on
the server, and their grouping differs between runs. The seed fixes the random draws, but not the
probabilities they are drawn from, and those carry the last-bit differences. A request with `n=1`
and a seed repeated in testing.

## What this means for experiments

Treat sampled outputs as reproducible when each one comes from its own seeded `n=1` request. Treat
greedy outputs as reproducible only when the requests ran one at a time. Both conditions cost
throughput, so decide per experiment whether exact reruns matter. For most evaluations, a mean
over enough samples is stable even when single outputs are not, and that is the better thing to
report.

In the method section, state:

- the sampling parameters, all of them, because the defaults come from the model and not from
  OpenAI
- whether the requests ran concurrently
- the vLLM version and the `serve check` report of the deployment

## Related pages

- [Get reproducible samples](../guides/get-reproducible-samples.md)
- [Validation results](../reference/validation-results.md)
