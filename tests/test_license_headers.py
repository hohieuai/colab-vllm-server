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

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_DIRS = ["src", "tests", "evals", "remote", "examples", ".devcontainer"]
SOURCES = sorted(
    p
    for d in SOURCE_DIRS
    for p in (ROOT / d).rglob("*")
    if p.suffix in (".py", ".sh") and "__pycache__" not in p.parts and "fixtures" not in p.parts
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_source_file_has_license_header(path):
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    assert lines[0] == "# Copyright 2026 The colab-vllm-server Authors"
    assert "Licensed under the Apache License, Version 2.0" in "\n".join(lines[:5])
