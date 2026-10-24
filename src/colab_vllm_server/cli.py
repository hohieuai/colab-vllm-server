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

"""serve: run an OpenAI-compatible vLLM endpoint on a Colab GPU.

serve resolve <preset>   print the resolved vllm serve command (offline)
serve up <preset>        provision the VM, start vLLM, open the SSH tunnel
serve env <preset>       print OPENAI_BASE_URL / OPENAI_API_KEY exports
serve chat <preset> MSG  send one chat message and print the answer
serve check <preset>     run the validation suite against the endpoint
serve connect <preset>   reopen the SSH tunnel to a running endpoint
serve status             list endpoints started from this checkout
serve down <preset>      stop the endpoint and release the VM
"""

import argparse
import dataclasses
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from colab_vllm_server.resolve import PORT, ROOT, ResolveError, resolve

RUNS_DIR = ROOT / "runs"
LOG_DIR = Path.home() / "colab-runs"  # read by colab-monitor's Logs page
READY_TIMEOUT_S = 45 * 60
DEFAULT_API_KEY = "vllm"
MAX_TUNNEL_RECONNECTS = 5


KILL_VLLM = """
import subprocess, time
subprocess.run(["pkill", "-9", "-f", "-i", "vllm"])
for _ in range(30):
    used = int(subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.split()[0])
    if used < 1024:
        break
    time.sleep(2)
print(f"GPU memory in use after cleanup: {used} MiB")
"""


def dotenv_get(key: str) -> str | None:
    """Value from the process environment, else from ROOT/.env.local."""
    if os.environ.get(key):
        return os.environ[key]
    path = ROOT / ".env.local"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.removeprefix("export ").strip() == key:
            return value.strip().strip("'\"") or None
    return None


def write_dotenv(path: Path, values: dict):
    """Set keys in a .env file in place, keeping every other line."""
    lines = path.read_text().splitlines() if path.exists() else []
    written = set()
    out = []
    for line in lines:
        name = line.partition("=")[0].removeprefix("export ").strip()
        if name in values:
            if name not in written:  # drop later copies of a repeated key
                out.append(f"{name}={values[name]}")
                written.add(name)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in values.items() if k not in written]
    path.write_text("\n".join(out) + "\n")
    path.chmod(0o600)


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def free_port(start: int) -> int:
    """First local port >= start that nothing is listening on."""
    for port in range(start, start + 100):
        if not port_in_use(port):
            return port
    sys.exit(f"[serve] no free local port in {start}-{start + 99}")


def colab_sessions() -> str:
    """Output of `colab sessions`; exits with instructions if not signed in.

    The Colab CLI documents `colab sessions` as the read-only sign-in check.
    Without credentials it prompts for an authorization code, so stdin is
    closed here to make that prompt fail instead of hang.
    """
    try:
        proc = subprocess.run(
            [colab_bin(), "sessions"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        sys.exit("[serve] `colab sessions` did not answer within 2 minutes; check your network")
    output = proc.stdout + proc.stderr
    if proc.returncode != 0 or "authorization code" in output.lower():
        sys.exit(
            "[serve] the colab CLI is not signed in. Run `colab sessions` in a terminal, "
            "complete the sign-in, and try again."
        )
    return output


def colab_bin() -> str:
    path = os.environ.get("COLAB_BIN") or shutil.which("colab")
    if not path:
        sys.exit(
            "[serve] the colab CLI is not installed. Install it with "
            "`uv tool install google-colab-cli`, or set COLAB_BIN to its path."
        )
    return path


def session_name(preset: str) -> str:
    return "vllm-" + preset.replace(".", "-")


def state_file(preset: str) -> Path:
    return RUNS_DIR / f"{preset}.state.json"


def load_state(preset: str) -> dict | None:
    f = state_file(preset)
    return json.loads(f.read_text()) if f.exists() else None


def save_state(preset: str, state: dict):
    RUNS_DIR.mkdir(exist_ok=True)
    f = state_file(preset)
    f.write_text(json.dumps(state, indent=1))
    f.chmod(0o600)  # holds the API key


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def command_line(pid: int) -> str:
    out = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def kill_group(pid: int | None, *markers: str):
    """SIGTERM the process group of `pid` if its command line has every marker.

    State files outlive reboots, and a saved PID can belong to an unrelated
    process by then. The markers make sure only a process this tool started
    is stopped.
    """
    if not alive(pid):
        return
    cmd = command_line(pid)
    if not all(m in cmd for m in markers):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass


def stop_tunnel(state: dict):
    kill_group(state.get("tunnel_pid"), "ssh", state["session"])


def stop_log_stream(state: dict):
    kill_group(state.get("exec_pid"), "exec", state["session"])


def served_models(port: int, api_key: str) -> list[str] | None:
    """Model ids from GET /v1/models, or None if the server does not answer."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return [m["id"] for m in json.load(resp).get("data", [])]
    except Exception:  # noqa: BLE001 - any failure means "not answering"
        return None


def cmd_resolve(args):
    r = resolve(args.preset, refresh=args.refresh)
    if args.json:
        print(json.dumps(dataclasses.asdict(r), indent=1))
        return
    print(f"# preset   {r.preset}")
    print(f"# recipe   {r.recipe} @ {r.recipe_sha[:10]}")
    print(f"# variant  {r.variant} ({r.precision}) -> {r.model_id}")
    print(f"# gpu      Colab {r.gpu} -> recipe profile {r.recipe_hw}")
    for w in r.warnings:
        print(f"# WARNING  {w}")
    print("\n# install\n" + r.install_script)
    print("\n# serve")
    print(r.command())


def open_tunnel(session: str, port: int, log) -> subprocess.Popen:
    proxy = f"{colab_bin()} ssh --proxy-mode -s {session}"
    return subprocess.Popen(
        [
            "ssh",
            "-N",
            "-o",
            f"ProxyCommand={proxy}",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-L",
            f"{port}:127.0.0.1:{PORT}",
            f"root@{session}",
        ],
        stdout=log,
        stderr=log,
        start_new_session=True,
    )


def cmd_up(args):
    r = resolve(args.preset, refresh=args.refresh)
    for w in r.warnings:
        print(f"[serve] WARNING: {w}")
    existing = load_state(r.preset)
    if existing and endpoint_ready(existing) and not args.restart:
        sys.exit(
            f"[serve] {r.preset} is already running; pass --restart to relaunch "
            f"on the same VM, or `serve down {r.preset}`"
        )

    colab = colab_bin()
    session = session_name(r.preset)
    reuse = f"[{session}]" in colab_sessions()
    if reuse and not args.restart:
        # Restarting would stop a server that may be healthy behind a dropped
        # tunnel, so it has to be asked for.
        sys.exit(
            f"[serve] the VM {session} already exists. If the endpoint stopped answering, "
            f"run `serve connect {r.preset}`. To relaunch the server on this VM, pass --restart. "
            f"To start over, run `serve down {r.preset}` first."
        )
    if existing:
        stop_tunnel(existing)
        stop_log_stream(existing)
        time.sleep(1)
    if args.port is None:
        previous = existing.get("port") if existing else None
        args.port = previous if previous and not port_in_use(previous) else free_port(PORT)
    elif port_in_use(args.port):
        sys.exit(f"[serve] local port {args.port} is in use; choose another with --port")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}-{r.preset}"
    run_dir.mkdir(parents=True)
    (run_dir / "resolved.json").write_text(json.dumps(dataclasses.asdict(r), indent=1))
    launch = (
        (ROOT / "remote" / "launch.py")
        .read_text()
        .replace("__RESOLVED_JSON__", json.dumps(dataclasses.asdict(r)))
    )
    launch_path = run_dir / "launch.py"
    launch_path.write_text(launch)

    if reuse:
        # The old launch cell keeps the kernel busy; restart it, then kill the
        # orphaned vLLM processes so the GPU is free. The venv on /content is
        # kept, so the relaunch skips the install.
        print(f"[serve] reusing VM {session}: restarting kernel and stopping old vLLM ...")
        subprocess.run([colab, "restart-kernel", "-s", session], check=True)
        subprocess.run(
            [colab, "exec", "-s", session, "--timeout", "120"],
            input=KILL_VLLM,
            text=True,
            check=True,
        )
    else:
        print(f"[serve] provisioning Colab {r.gpu} as session {session} ...")
        new = [colab, "new", "-s", session, "--gpu", r.accelerator]
        if r.high_mem or args.high_mem:
            new.append("--high-mem")
        if subprocess.run(new, check=False).returncode != 0:
            sys.exit(
                f"[serve] `colab new` failed. A 400 error means this account has no quota or "
                f"entitlement for {r.accelerator}"
                + (" with --high-mem" if r.high_mem or args.high_mem else "")
                + ". Check `colab usage` and your Colab plan."
            )

    # The server is reachable only through the SSH tunnel, so the key is a
    # fixed placeholder by default; set VLLM_API_KEY to use a real one.
    api_key = dotenv_get("VLLM_API_KEY") or DEFAULT_API_KEY
    LOG_DIR.mkdir(exist_ok=True)
    server_log = LOG_DIR / f"{session}.log"
    exec_cmd = [
        colab,
        "exec",
        "-s",
        session,
        "-f",
        str(launch_path),
        "--timeout",
        str(7 * 24 * 3600),
        "--env",
        f"VLLM_API_KEY={api_key}",
    ]
    hf_token = dotenv_get("HF_TOKEN")
    if hf_token:
        exec_cmd += ["--env", f"HF_TOKEN={hf_token}"]
    with open(server_log, "w") as log:
        exec_proc = subprocess.Popen(
            exec_cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    tunnel_log = open(run_dir / "tunnel.log", "w")  # noqa: SIM115 - outlives this function in ssh
    tunnel = open_tunnel(session, args.port, tunnel_log)

    state = {
        "preset": r.preset,
        "session": session,
        "gpu": r.gpu,
        "port": args.port,
        "api_key": api_key,
        "exec_pid": exec_proc.pid,
        "tunnel_pid": tunnel.pid,
        "run_dir": str(run_dir),
        "log": str(server_log),
        "served_model_name": r.served_model_name,
        "started": stamp,
    }
    save_state(r.preset, state)
    print(f"[serve] server log: {server_log}")

    base = f"http://127.0.0.1:{args.port}"
    deadline = time.time() + READY_TIMEOUT_S
    reconnects = 0
    while time.time() < deadline:
        if exec_proc.poll() is not None:
            print(server_log.read_text()[-3000:])
            sys.exit(
                f"[serve] `colab exec` ended before the server was ready (exit "
                f"{exec_proc.returncode}). If the log ends with 'Connection was lost', the server "
                f"may still be starting on the VM: wait, then run `serve connect {r.preset}`. "
                f"Otherwise fix the error above. The VM is still running; release it with "
                f"`serve down {r.preset}`."
            )
        if tunnel.poll() is not None:
            print((run_dir / "tunnel.log").read_text()[-2000:])
            reconnects += 1
            if reconnects > MAX_TUNNEL_RECONNECTS:
                sys.exit(
                    f"[serve] the tunnel failed {MAX_TUNNEL_RECONNECTS} times; see the output "
                    f"above. The VM is still running: fix the cause, then run "
                    f"`serve connect {r.preset}`, or release it with `serve down {r.preset}`."
                )
            print("[serve] tunnel dropped, reconnecting ...")
            # The VM accepts one `colab ssh` connection; let the old one close.
            time.sleep(5)
            tunnel = open_tunnel(session, args.port, tunnel_log)
            state["tunnel_pid"] = tunnel.pid
            save_state(r.preset, state)
        if endpoint_ready(state):
            break
        time.sleep(10)
    else:
        sys.exit(f"[serve] not ready after {READY_TIMEOUT_S // 60} min; see {server_log}")

    print(f"""
[serve] ready: {r.served_model_name} on Colab {r.gpu}
  export OPENAI_BASE_URL={base}/v1
  export OPENAI_API_KEY={api_key}
  model name: {r.served_model_name}
[serve] stop with: serve down {r.preset}""")


def endpoint_ready(state: dict) -> bool:
    """True if this preset's server answers on its port.

    Checks the served model name too, so a server of another preset that now
    holds the same local port is not taken for this one.
    """
    models = served_models(state["port"], state["api_key"])
    return models is not None and state["served_model_name"] in models


def cmd_connect(args):
    """Reopen the SSH tunnel, for example after the computer slept."""
    state = load_state(args.preset)
    if not state:
        sys.exit(f"[serve] {args.preset} is not running; `serve up {args.preset}` first")
    # A runtime accepts one `colab ssh` connection at a time (HTTP 429), so the
    # old tunnel must be gone before the new one connects.
    stop_tunnel(state)
    time.sleep(2)
    tunnel_log = open(Path(state["run_dir"]) / "tunnel.log", "a")  # noqa: SIM115 - outlives this function in ssh
    tunnel = open_tunnel(state["session"], state["port"], tunnel_log)
    state["tunnel_pid"] = tunnel.pid
    save_state(args.preset, state)
    for _ in range(30):
        if endpoint_ready(state):
            print(f"[serve] {args.preset} reachable at http://127.0.0.1:{state['port']}/v1")
            return
        if tunnel.poll() is not None:
            break
        time.sleep(2)
    sys.exit(
        f"[serve] the tunnel is open but the server does not answer. Check the end of "
        f"{state['log']}, and run `colab sessions` to confirm the VM still exists."
    )


def cmd_down(args):
    state = load_state(args.preset)
    session = state["session"] if state else session_name(args.preset)
    if state:
        stop_tunnel(state)
        stop_log_stream(state)
    # `colab stop` exits 0 for a session that is already gone, and may fail
    # for other reasons, so the session list is the evidence.
    subprocess.run([colab_bin(), "stop", "-s", session], check=False)
    if f"[{session}]" in colab_sessions():
        sys.exit(
            f"[serve] the VM {session} is still running after `colab stop`, and still uses "
            f"compute units. Run `serve down {args.preset}` again, or `colab stop -s {session}`."
        )
    state_file(args.preset).unlink(missing_ok=True)
    print(f"[serve] {args.preset} stopped and VM released")


def cmd_status(args):
    found = False
    for f in sorted(RUNS_DIR.glob("*.state.json")) if RUNS_DIR.exists() else []:
        s = json.loads(f.read_text())
        found = True
        base = f"http://127.0.0.1:{s['port']}"
        # The API answer is the only reliable signal: the server keeps running
        # on the VM when the local `colab exec` that streams its log has ended.
        print(
            f"{s['preset']:<28} Colab {s['gpu']:<8} session={s['session']} "
            f"api={'ready' if endpoint_ready(s) else 'unreachable'} "
            f"tunnel={'up' if alive(s['tunnel_pid']) else 'down'} "
            f"log={'streaming' if alive(s['exec_pid']) else 'stopped'} {base}/v1"
        )
    if not found:
        print("no endpoints running")


def cmd_env(args):
    state = load_state(args.preset)
    if not state:
        sys.exit(f"[serve] {args.preset} is not running; `serve up {args.preset}` first")
    if args.write:
        values = {
            "OPENAI_BASE_URL": f"http://127.0.0.1:{state['port']}/v1",
            "OPENAI_API_KEY": state["api_key"],
        }
        write_dotenv(ROOT / ".env.local", values)
        print(f"[serve] updated {ROOT / '.env.local'} for {args.preset}")
        return
    prefix = "" if args.dotenv else "export "
    print(f"{prefix}OPENAI_BASE_URL=http://127.0.0.1:{state['port']}/v1")
    print(f"{prefix}OPENAI_API_KEY={state['api_key']}")
    hf_token = dotenv_get("HF_TOKEN")
    if args.dotenv and hf_token:
        print(f"HF_TOKEN={hf_token}")


def cmd_chat(args):
    """Send one message and stream the answer: a quick test of a running endpoint."""
    from openai import OpenAI

    state = load_state(args.preset)
    if not state:
        sys.exit(f"[serve] {args.preset} is not running; `serve up {args.preset}` first")
    if not endpoint_ready(state):
        sys.exit(
            f"[serve] {args.preset} does not answer. Run `serve status`, and "
            f"`serve connect {args.preset}` if the tunnel is down."
        )
    client = OpenAI(base_url=f"http://127.0.0.1:{state['port']}/v1", api_key=state["api_key"])
    started = time.time()
    stream = client.chat.completions.create(
        model=state["served_model_name"],
        messages=[{"role": "user", "content": args.message}],
        max_tokens=args.max_tokens or (8192 if args.think else 1024),
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": args.think}},
    )
    usage, finish, thinking, answering = None, None, False, False
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if reasoning:
            if not thinking:
                print("[thinking] ", end="", file=sys.stderr)
                thinking = True
            print(reasoning, end="", flush=True, file=sys.stderr)
        content = delta.content or ""
        if not answering:
            # The model often opens its answer with blank lines after thinking.
            content = content.lstrip()
        if content:
            if thinking:
                print(file=sys.stderr, flush=True)
                thinking = False
            answering = True
            print(content, end="", flush=True)
        finish = chunk.choices[0].finish_reason or finish
    print(flush=True)
    tokens = usage.completion_tokens if usage else "?"
    print(
        f"[serve] {state['served_model_name']}: {tokens} tokens in {time.time() - started:.1f} s, "
        f"finish_reason={finish}",
        file=sys.stderr,
    )
    if finish == "length":
        print("[serve] the answer was cut off; raise --max-tokens", file=sys.stderr)


def cmd_check(args):
    from colab_vllm_server.check import run_checks

    state = load_state(args.preset)
    if not state:
        sys.exit(f"[serve] {args.preset} is not running; `serve up {args.preset}` first")
    only = args.only.split(",") if args.only else None
    run_checks(state, only=only, accuracy=args.accuracy)


def main():
    p = argparse.ArgumentParser(
        prog="serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("resolve", help="print the resolved vllm serve command (offline)")
    sp.add_argument("preset")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--refresh", action="store_true", help="re-fetch recipes.vllm.ai")
    sp.set_defaults(func=cmd_resolve)

    sp = sub.add_parser("up", help="provision Colab and start the endpoint")
    sp.add_argument("preset")
    sp.add_argument(
        "--port", type=int, default=None, help=f"local port (default: first free from {PORT})"
    )
    sp.add_argument("--high-mem", action="store_true")
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument(
        "--restart",
        action="store_true",
        help="relaunch on the same VM if the endpoint is already running",
    )
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("down", help="stop the endpoint and release the VM")
    sp.add_argument("preset")
    sp.set_defaults(func=cmd_down)

    sp = sub.add_parser("env", help="print OPENAI_BASE_URL and OPENAI_API_KEY exports")
    sp.add_argument("preset")
    sp.add_argument(
        "--dotenv",
        action="store_true",
        help="print KEY=value lines for a .env file instead of shell exports",
    )
    sp.add_argument(
        "--write",
        action="store_true",
        help="set OPENAI_BASE_URL and OPENAI_API_KEY in .env.local, keeping other lines",
    )
    sp.set_defaults(func=cmd_env)

    sp = sub.add_parser("check", help="run the validation suite against the endpoint")
    sp.add_argument("preset")
    sp.add_argument("--only", help="comma-separated check names")
    sp.add_argument("--accuracy", action="store_true", help="also run GSM8K (100 questions)")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("chat", help="send one chat message and print the answer")
    sp.add_argument("preset")
    sp.add_argument(
        "message",
        nargs="?",
        default="Hello! Introduce yourself in one sentence.",
        help="message to send (default: a short greeting)",
    )
    sp.add_argument("--think", action="store_true", help="turn thinking on and show it")
    sp.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="answer length limit (default: 1024, or 8192 with --think)",
    )
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("connect", help="reopen the SSH tunnel to a running endpoint")
    sp.add_argument("preset")
    sp.set_defaults(func=cmd_connect)

    sp = sub.add_parser("status", help="list endpoints started from this checkout")
    sp.set_defaults(func=cmd_status)

    args = p.parse_args()
    try:
        args.func(args)
    except ResolveError as e:
        sys.exit(f"[serve] cannot resolve: {e}")


if __name__ == "__main__":
    main()
