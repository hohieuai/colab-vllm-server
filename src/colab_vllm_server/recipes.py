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

"""Fetch and cache the recipes.vllm.ai JSON API.

The site publishes, per recipe, a model document (`/<hf_id>.json`) and fully
synthesized commands per (variant checkpoint, hardware) pair
(`/<variant_hf_id>/hw/<hw>.json`). We reuse the synthesized argv instead of
re-implementing their override/dedupe logic.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://recipes.vllm.ai"
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "recipes"


class RecipeNotFound(Exception):
    pass


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def fetch(path: str, refresh: bool = False) -> dict:
    """GET `BASE_URL + path`, caching the body under cache/recipes/.

    Falls back to the cached copy when the network or the server fails. A 404
    raises RecipeNotFound.
    """
    cache_file = CACHE_DIR / path.lstrip("/")
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())
    try:
        body = _download(BASE_URL + path)
        data = json.loads(body)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RecipeNotFound(BASE_URL + path) from e
        if cache_file.exists():  # a server error is no reason to drop a good copy
            return json.loads(cache_file.read_text())
        raise
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        raise
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data, indent=1))
    return data


def model(hf_id: str, refresh: bool = False) -> dict:
    return fetch(f"/{hf_id}.json", refresh)


def render(variant_hf_id: str, hw: str, refresh: bool = False) -> dict:
    return fetch(f"/{variant_hf_id}/hw/{hw}.json", refresh)
