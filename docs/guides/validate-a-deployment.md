# Validate a deployment

An endpoint can answer requests and still be wrong in ways that corrupt an experiment: reasoning
that leaks into the answer, tool calls that fail to parse, or outputs that change with server
load. This guide shows how to test a deployment with `serve check` before you use it.

## Run the checks

With the endpoint running, run:

```bash
uv run serve check qwen3.8-27b-a100-80g
```

The run takes about 5 minutes. Each check prints `PASS`, `WARN`, or `FAIL`:

```
[check] qwen3.8-27b-a100-80g: Qwen/Qwen3.8-27B on Colab A100_80G, frontend=python
  PASS  auth                     no-key -> HTTP 401; served models ['Qwen/Qwen3.8-27B']
  PASS  greedy_sanity            5/5 correct
  ...
[check] 14 pass, 2 warn, 0 fail
[check] report: runs/20261024-144007-qwen3.8-27b-a100-80g/check-20261024-154432.json
```

To add an accuracy check on 100 GSM8K questions, add `--accuracy`. To run only some checks, list
them with `--only`:

```bash
uv run serve check qwen3.8-27b-a100-80g --accuracy
uv run serve check qwen3.8-27b-a100-80g --only tool_calls,structured_json
```

## Act on the result

- **FAIL:** don't use the deployment for that feature. Read the check's detail line, then see
  [Troubleshooting](../troubleshooting.md).
- **WARN:** the feature works with a limit you must account for. The two warnings on the BF16
  deployment are about reproducibility. [Get reproducible samples](get-reproducible-samples.md)
  shows how to work with them.
- **PASS:** the property held in this run.

## Keep the report with your results

The report is a JSON file in the run's folder. It holds the resolved command, the recipe commit,
and every check result. Save its path next to your experiment's results, so a reader can see the
exact server behind each number.

## Run the checks again after a change

Run `serve check` again after any change that can change outputs:

- a new preset or GPU
- a new vLLM version, because presets install the latest release unless `vllm_version` is set
- a recipe update, after `serve resolve --refresh`

## Related pages

- [Validation results](../reference/validation-results.md): what each check tests, and the
  results of the BF16 deployment
- [Command line](../reference/command-line.md#serve-check)
