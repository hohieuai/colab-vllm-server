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

"""The commands that users run to get colab-vllm-server check out the current release."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
USER_DOCS = ["README.md", "docs/quickstart.md", "skills/colab-vllm-server/SKILL.md"]
CLONE = re.compile(r"git clone(?: --branch (\S+))? https://github\.com/hohieuai/colab-vllm-server")


def test_user_docs_clone_this_release():
    """A release PR moves every pin. CONTRIBUTING.md, for contributors, keeps main."""
    for doc in USER_DOCS:
        clones = CLONE.findall((ROOT / doc).read_text())
        assert clones, f"{doc} has no clone command"
        assert all(branch == f"v{VERSION}" for branch in clones), f"{doc}: {clones}"
