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

"""Run every test offline against recipe documents saved in tests/fixtures."""

import urllib.error
from pathlib import Path

import pytest

from colab_vllm_server import recipes

FIXTURES = Path(__file__).parent / "fixtures" / "recipes"


@pytest.fixture(autouse=True)
def offline_recipes(monkeypatch):
    def no_network(*args, **kwargs):
        raise urllib.error.URLError("network disabled in tests")

    monkeypatch.setattr(recipes, "CACHE_DIR", FIXTURES)
    monkeypatch.setattr(recipes, "_download", no_network)


@pytest.fixture
def preset(tmp_path):
    """Write a preset file from keyword arguments and return its path."""

    def write(name="test", **fields):
        lines = []
        for key, value in fields.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                lines += [
                    f"  {k}: {str(v).lower() if isinstance(v, bool) else v}"
                    for k, v in value.items()
                ]
            else:
                lines.append(f"{key}: {value}")
        path = tmp_path / f"{name}.yaml"
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    return write
