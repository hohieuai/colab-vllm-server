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

# Runs INSIDE the Colab kernel via `colab exec -f`. The local `serve up`
# command writes a copy of this file with RESOLVED filled in.
#
# vLLM runs as a foreground child of this cell on purpose: Colab keeps a VM
# alive based on kernel activity, so a busy cell is what keeps the endpoint
# up. Interrupting the cell (or `serve down`) stops the server.

import hashlib
import json
import os
import shutil
import subprocess
import sys

RESOLVED = json.loads(r"""__RESOLVED_JSON__""")
WORKDIR = "/content/colab-vllm-server"


def log(msg):
    print(f"[colab-vllm-server] {msg}", flush=True)


def stream(cmd, **kwargs):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        **kwargs,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def check_gpu():
    # Full nvidia-smi first, so every run log records the card, driver and
    # CUDA version the results were produced on.
    print(
        subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout,
        flush=True,
    )
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    name, mem_mib, driver = [s.strip() for s in out.splitlines()[0].split(",")]
    gb = int(mem_mib) / 1024
    log(f"GPU: {name}, {gb:.0f} GiB, driver {driver}")
    # Colab can hand out a smaller card than requested (e.g. the 40 GB A100
    # when the 80 GB High-RAM shape was asked for); fail before downloading.
    if gb + 2 < RESOLVED["vram_gb"]:
        raise SystemExit(
            f"expected ~{RESOLVED['vram_gb']} GB for {RESOLVED['gpu']}, got {gb:.0f} GiB"
        )


def install():
    venv = os.path.join(WORKDIR, ".venv")
    vllm_bin = os.path.join(venv, "bin", "vllm")
    # The stamp records the install script of the last install that finished.
    # A missing or different stamp means the venv is partial or outdated.
    stamp = os.path.join(venv, ".install-script.sha256")
    digest = hashlib.sha256(RESOLVED["install_script"].encode()).hexdigest()
    installed = ""
    if os.path.exists(stamp):
        with open(stamp) as f:
            installed = f.read()
    if os.path.exists(vllm_bin) and installed == digest:
        log("vLLM venv already installed with this install script, skipping install")
        return vllm_bin
    if os.path.exists(venv):
        log("removing the previous venv: its install did not finish or used another script")
        shutil.rmtree(venv)
    if shutil.which("uv") is None:
        log("installing uv")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    log("installing vLLM (recipe pip path):\n" + RESOLVED["install_script"])
    script = "set -euo pipefail\n" + RESOLVED["install_script"]
    # Colab exports UV_SYSTEM_PYTHON=1, which makes `uv pip install` ignore the
    # activated venv and overwrite the kernel's own packages under /usr.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UV_SYSTEM_PYTHON", "UV_PYTHON", "VIRTUAL_ENV")
    }
    if stream(["bash", "-c", script], cwd=WORKDIR, env=env) != 0:
        raise SystemExit("install failed")
    if not os.path.exists(vllm_bin):
        raise SystemExit(f"install finished but {vllm_bin} is missing")
    with open(stamp, "w") as f:
        f.write(digest)
    return vllm_bin


def serve(vllm_bin):
    env = dict(os.environ)
    env.update(RESOLVED["env"])
    env["PATH"] = os.path.join(WORKDIR, ".venv", "bin") + ":" + env.get("PATH", "")
    if not env.get("VLLM_API_KEY"):
        log("WARNING: VLLM_API_KEY not set; endpoint is unauthenticated")
    argv = [vllm_bin] + RESOLVED["argv"][1:]
    version = subprocess.run(
        [
            os.path.join(WORKDIR, ".venv", "bin", "python"),
            "-c",
            "import vllm; print(vllm.__version__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    log(f"vLLM {version}; recipe {RESOLVED['recipe']}@{RESOLVED['recipe_sha'][:10]}")
    log("starting: " + " ".join(argv))
    code = stream(argv, cwd=WORKDIR, env=env)
    raise SystemExit(f"vllm serve exited with code {code}")


os.makedirs(WORKDIR, exist_ok=True)
check_gpu()
serve(install())
