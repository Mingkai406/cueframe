#!/usr/bin/env python3
"""Controlled visual perturbations of a CC0 photo; not a recorded human interaction."""

from PIL import Image
from pathlib import Path
import argparse, json

p = argparse.ArgumentParser()
p.add_argument("--out", default="runs/fixture")
p.add_argument("--fps", type=int, default=30)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
base = Image.open(Path(__file__).resolve().parents[1] / "examples/coffee.png").convert(
    "RGB"
)
for n in range(a.fps * 8):
    t = n / a.fps
    if t < 2:
        im = base.copy()
        phase = "stationary"
    elif t < 3:
        im = Image.new("RGB", base.size, (235, 234, 230))
        im.paste(base, (int((t - 2) * 100), 0))
        phase = "translated"
    elif t < 4:
        im = Image.new("RGB", base.size, (235, 234, 230))
        phase = "blank"
    elif t < 6:
        im = Image.new("RGB", base.size, (235, 234, 230))
        tile = base.resize((300, 200))
        im.paste(tile, (0, 100))
        im.paste(tile, (300, 100))
        phase = "duplicated"
    else:
        im = base.copy()
        phase = "returned"
    im.save(out / f"{n:05d}.jpg", quality=90)
(out / "manifest.json").write_text(
    json.dumps(
        {
            "kind": "controlled image transformations",
            "source": "CC0 coffee by Rachel Michetti / scikit-image",
            "fps": a.fps,
            "frames": a.fps * 8,
            "phases": [
                [0, 2, "stationary"],
                [2, 3, "translated"],
                [3, 4, "blank"],
                [4, 6, "duplicated"],
                [6, 8, "returned"],
            ],
        },
        indent=2,
    )
)
print(out)
