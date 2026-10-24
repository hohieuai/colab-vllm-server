#!/usr/bin/env bash
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

# Set up the dev container: uv, the project environment, the Colab CLI, and
# a copy of the host's SSH keys with the permissions ssh requires.
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync
uv tool install google-colab-cli

# The host's ~/.ssh is mounted read-only at ~/.ssh-host. ssh refuses keys that
# other users can read, so copy them with private permissions.
if [ -d "$HOME/.ssh-host" ]; then
  mkdir -p "$HOME/.ssh"
  cp -r "$HOME/.ssh-host/." "$HOME/.ssh/"
  chmod 700 "$HOME/.ssh"
  find "$HOME/.ssh" -type f -name "id_*" ! -name "*.pub" -exec chmod 600 {} +
fi

sudo chown -R "$(id -u):$(id -g)" "$HOME/.config/colab-cli"

echo "Done. Sign in to Colab once with: colab sessions"
