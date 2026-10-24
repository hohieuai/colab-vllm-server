# How colab-vllm-server works

colab-vllm-server turns a short preset into a running vLLM server on a Colab VM and makes it reachable
from your computer. This page explains the parts and why each one is built the way it is.

## From preset to command

A vLLM command for a large model has many flags that interact: parsers for reasoning and tool
calls, attention and MoE backends, memory settings, and speculative decoding. The right values
depend on the model, the checkpoint, and the GPU. Writing them by hand is where most broken
deployments start.

[vLLM Recipes](https://recipes.vllm.ai) already maintains these values. Each recipe page
publishes a JSON document for every pair of checkpoint and hardware profile, with the complete
argv. colab-vllm-server uses that argv instead of building one. The preset only names the recipe, the
variant, and the GPU, and adds the few settings Colab forces: one GPU, a loopback address, and a
fixed port.

The alternative was to read the recipe's raw fields and repeat the site's own logic for combining
them. That logic has layers for hardware generation, exact chip, variant, strategy, and feature
mode, each with its own precedence rule. A copy of it would drift from the site, and a drift would
change commands without a visible error. Taking the published argv keeps colab-vllm-server aligned with
what the recipe maintainers test.

The cost shows on GPUs the recipes don't list. The recipes have no profile for the A100 or the L4,
so those GPUs take the H100 command and remove what the recipe meant only for Hopper. That
adaptation repeats a small part of the site's logic, and its result is untested by the recipe
maintainers. This is why `serve resolve` warns on every adapted change, and why borrowed
deployments need `serve check`. [GPUs](../reference/gpus.md#borrowed-profiles) lists the steps.

## Checks before the VM starts

A Colab GPU costs compute units from the moment the VM starts, and the first launch of a large
model takes 15 minutes. A mistake found after the download wastes both. The resolver therefore
checks everything it can from the recipe before it asks Colab for a VM: memory, eligible
hardware, precision support, and node count.

The memory check uses the recipe's `vram_minimum_gb`, which is a formula estimate. For some
models it is far from the truth: gpt-oss-120b is estimated at 96 GB, but its checkpoint is 65 GB
and runs on one 80 GB GPU. A preset can replace the estimate, and the resolver records that it
did.

## Running on the VM

Colab VMs have no Docker, so `launch.py` installs vLLM with the recipe's pip command into a
virtual environment. Colab sets `UV_SYSTEM_PYTHON=1`, which makes `uv pip install` write into the
system Python and replace the kernel's own packages. `launch.py` clears that variable for the
install.

Colab keeps a VM alive while its notebook kernel is busy. `launch.py` therefore runs vLLM as a
child of the running cell and streams its output. The cell never finishes while the server runs,
so the kernel stays busy and the VM stays up. A server started in the background and left alone
would leave the kernel idle, and Colab would reclaim the VM.

The cell runs on the VM, and the local `colab exec` only watches it. If the connection drops, for
example when the computer sleeps, the local process ends with `Connection was lost`, but the cell
and the server keep running. That is why `serve status` decides whether an endpoint works from the
API answer, not from the local process.

## The tunnel

The server listens on `127.0.0.1` on the VM, so nothing outside the VM can reach it directly.
The `colab ssh --proxy-mode` command carries SSH over Colab's own connection to the VM, and
`ssh -L` forwards a local port through it. Your computer talks to `127.0.0.1:8000`, and the
request arrives at the server on the VM.

This avoids a public URL. A service such as a public tunnel would expose the server to anyone
with the address, and the API key would be the only protection. With the SSH tunnel, the key is
a second layer. That is why the default key can be the placeholder `vllm`.

The tunnel has two limits from Colab. First, a runtime accepts one `colab ssh` connection at a
time, so the tunnel holds the VM's only SSH slot. Second, the token that authenticates the
connection expires after an hour. An open connection keeps working past that hour because Colab
checks the token only when a connection starts. A new connection, such as the one
`serve connect` opens, gets a fresh token from `colab ssh`.

## Why presets use the Python frontend

vLLM has two HTTP frontends. The recipes select the newer Rust frontend by default. On vLLM 0.32,
it rejects `n` greater than 1 and logs that structured-output settings have no effect. In testing,
1 of 4 schema-constrained answers came back cut off. Both limits matter for research: `n` samples
are how pass@k and preference pairs are made, and schema enforcement is what makes LLM-as-judge
output parse.

The Python frontend passed every check at the same throughput. Until the Rust frontend reaches
parity, every preset sets `frontend: python`. When a new vLLM release changes this, `serve check`
shows it.

## What gets recorded

A number in a paper is only as good as the record of the server that produced it. Every run
saves the resolved command, the recipe commit, the install script, and the script that ran on the
VM. The server log starts with the `nvidia-smi` output and the vLLM version. `serve check`
reports sit in the same folder. Together, they answer the questions a reviewer asks: which
checkpoint, which flags, which version, and on which card.
