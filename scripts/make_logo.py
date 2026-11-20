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

"""Write the colab-vllm-server logo to docs/images.

    uv run python scripts/make_logo.py

The mark is a paragraph of streamed text drawn as dots, one dot per token. Lines of different
lengths read as text. The newest tokens are larger, and the last one, the cursor, is the largest,
so the trail grows toward the point where the model is writing. It shares its dots, its trail and
its tile with the Colab Monitor logo.

It writes `logo-light.svg` and `logo-dark.svg`, the mark alone for GitHub's light and dark
themes, and `icon.svg`, the mark on a dark rounded tile.
"""

import pathlib

IMAGES = pathlib.Path(__file__).resolve().parent.parent / "docs" / "images"
LINES = (6, 4, 6, 5, 2)  # tokens on each line of the paragraph
PITCH = 7.2  # distance between neighbouring tokens
DIM = 0.9  # radius of an old token
LIT = 3.2  # radius of the cursor
TRAIL = 8  # tokens the trail grows over, ending at the cursor
GROUND = "#0a0a0a"


def num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def tokens() -> list[tuple[float, float, float]]:
    """Each token's centre and radius, centred on (32, 32) by its outer edges."""
    count = sum(LINES)
    dots, index = [], 0
    for row, length in enumerate(LINES):
        for col in range(length):
            age = count - 1 - index  # 0 is the cursor
            glow = max(0.0, 1 - age / TRAIL)
            dots.append((col * PITCH, row * PITCH, DIM + (LIT - DIM) * glow))
            index += 1
    left = min(x - r for x, _, r in dots)
    right = max(x + r for x, _, r in dots)
    top = min(y - r for _, y, r in dots)
    bottom = max(y + r for _, y, r in dots)
    dx, dy = 32 - (left + right) / 2, 32 - (top + bottom) / 2
    return [(x + dx, y + dy, r) for x, y, r in dots]


def mark(ink: str) -> str:
    circles = "".join(f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r)}"/>' for x, y, r in tokens())
    return f'<g fill="{ink}">{circles}</g>'


def bare(ink: str) -> str:
    """The mark on a transparent ground, in a square cropped to the dots."""
    half = max(max(abs(x - 32) + r, abs(y - 32) + r) for x, y, r in tokens()) + 1
    box = f"{num(32 - half)} {num(32 - half)} {num(2 * half)} {num(2 * half)}"
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}">{mark(ink)}</svg>'


def icon() -> str:
    """The mark on a dark rounded tile, with a faint edge so the tile shows on black."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        f'<rect width="64" height="64" rx="15" fill="{GROUND}"/>'
        '<rect x="0.5" y="0.5" width="63" height="63" rx="14.5" fill="none" stroke="#fff" '
        'stroke-opacity=".16"/>'
        f"{mark('#ffffff')}</svg>"
    )


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    files = {"logo-light.svg": bare(GROUND), "logo-dark.svg": bare("#ffffff"), "icon.svg": icon()}
    for name, svg in files.items():
        (IMAGES / name).write_text(svg + "\n")
        print(f"wrote docs/images/{name}")


if __name__ == "__main__":
    main()
