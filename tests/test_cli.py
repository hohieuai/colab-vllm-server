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

"""`serve` lifecycle safeguards, with the colab CLI replaced by stand-ins."""

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from colab_vllm_server import cli
from colab_vllm_server.check import run_checks
from colab_vllm_server.resolve import Resolved

SESSION = "vllm-qwen3-8-27b-a100-80g"


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(cli, "colab_bin", lambda: "true")
    return tmp_path


def run_serve(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["serve", *argv])
    cli.main()


def test_kill_group_spares_a_process_without_the_markers():
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        cli.kill_group(proc.pid, "ssh", SESSION)
        time.sleep(0.2)
        assert proc.poll() is None
        cli.kill_group(proc.pid, "sleep")
        assert proc.wait(timeout=5) is not None
    finally:
        proc.kill()


def test_write_dotenv_updates_every_copy_of_a_key(tmp_path):
    env = tmp_path / ".env.local"
    env.write_text("OPENAI_API_KEY=old\nHF_TOKEN=keep\nOPENAI_API_KEY=stale\n")
    cli.write_dotenv(env, {"OPENAI_API_KEY": "vllm", "OPENAI_BASE_URL": "http://x/v1"})
    assert env.read_text() == "OPENAI_API_KEY=vllm\nHF_TOKEN=keep\nOPENAI_BASE_URL=http://x/v1\n"


def test_down_keeps_the_state_when_the_vm_is_still_listed(runs, monkeypatch):
    cli.save_state("qwen3.8-27b-a100-80g", {"session": SESSION, "port": 8000, "api_key": "vllm"})
    monkeypatch.setattr(cli, "colab_sessions", lambda: f"[{SESSION}] gpu-a100 | Hardware: A100")
    with pytest.raises(SystemExit, match="still running"):
        run_serve(monkeypatch, "down", "qwen3.8-27b-a100-80g")
    assert cli.state_file("qwen3.8-27b-a100-80g").exists()


def test_down_removes_the_state_when_the_vm_is_gone(runs, monkeypatch, capsys):
    cli.save_state("qwen3.8-27b-a100-80g", {"session": SESSION, "port": 8000, "api_key": "vllm"})
    monkeypatch.setattr(cli, "colab_sessions", lambda: "[colab] No active sessions found.")
    run_serve(monkeypatch, "down", "qwen3.8-27b-a100-80g")
    assert not cli.state_file("qwen3.8-27b-a100-80g").exists()
    assert "VM released" in capsys.readouterr().out


def test_up_refuses_to_restart_an_existing_vm_without_restart(runs, monkeypatch):
    monkeypatch.setattr(cli, "colab_sessions", lambda: f"[{SESSION}] gpu-a100 | Hardware: A100")
    with pytest.raises(SystemExit, match="already exists"):
        run_serve(monkeypatch, "up", "qwen3.8-27b-a100-80g")


def test_up_passes_idle_timeout_zero_to_colab_exec(runs, monkeypatch):
    """Long-running launch must use --timeout 0 (no idle limit on current colab CLI)."""
    resolved = Resolved(
        preset="qwen3.8-27b-a100-80g",
        recipe="Qwen/Qwen3.8-27B",
        recipe_sha="abc",
        variant="default",
        model_id="Qwen/Qwen3.8-27B",
        precision="bf16",
        gpu="A100_80G",
        accelerator="A100",
        high_mem=True,
        recipe_hw="h100",
        exact_hw=False,
        vram_gb=80,
        served_model_name="Qwen/Qwen3.8-27B",
        argv=["vllm", "serve", "Qwen/Qwen3.8-27B"],
        env={},
        install_script="true",
        warnings=[],
    )
    monkeypatch.setattr(cli, "resolve", lambda preset, refresh=False: resolved)
    monkeypatch.setattr(cli, "colab_sessions", lambda: "[colab] No active sessions found.")
    monkeypatch.setattr(cli, "colab_bin", lambda: "colab")
    monkeypatch.setattr(cli, "free_port", lambda start: 18000)
    monkeypatch.setattr(cli, "dotenv_get", lambda key: None)
    monkeypatch.setattr(cli, "LOG_DIR", runs)
    monkeypatch.setattr(
        cli, "open_tunnel", lambda session, port, log: SimpleNamespace(pid=99, poll=lambda: None)
    )
    monkeypatch.setattr(
        cli,
        "endpoint_ready",
        lambda state: True,  # skip the ready wait after exec+tunnel start
    )

    captured = {}

    def fake_run(cmd, **kwargs):
        captured.setdefault("run", []).append(list(cmd))
        return SimpleNamespace(returncode=0)

    def fake_popen(cmd, **kwargs):
        captured["popen"] = list(cmd)
        return SimpleNamespace(pid=42, poll=lambda: None)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)

    run_serve(monkeypatch, "up", "qwen3.8-27b-a100-80g")

    cmd = captured["popen"]
    assert cmd[0:4] == ["colab", "exec", "-s", SESSION]
    assert cmd[cmd.index("-f") + 1].endswith("/launch.py")
    assert Path(cmd[cmd.index("-f") + 1]).is_file()
    assert cmd[cmd.index("--timeout") + 1] == "0"
    assert f"VLLM_API_KEY={cli.DEFAULT_API_KEY}" in cmd
    state = cli.load_state("qwen3.8-27b-a100-80g")
    assert state["port"] == 18000
    assert state["exec_pid"] == 42
    assert state["tunnel_pid"] == 99
    assert any(c[:2] == ["colab", "new"] and "--gpu" in c and "A100" in c for c in captured["run"])


def test_endpoint_ready_requires_the_presets_model(monkeypatch):
    monkeypatch.setattr(cli, "served_models", lambda port, key: ["other/model"])
    assert not cli.endpoint_ready({"port": 8000, "api_key": "vllm", "served_model_name": "a/b"})
    monkeypatch.setattr(cli, "served_models", lambda port, key: ["a/b"])
    assert cli.endpoint_ready({"port": 8000, "api_key": "vllm", "served_model_name": "a/b"})


def test_check_rejects_an_unknown_check_name_before_running():
    with pytest.raises(SystemExit, match="unknown check"):
        run_checks({"port": 1, "api_key": "x", "served_model_name": "m"}, only=["tool_call"])
