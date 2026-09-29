#!/usr/bin/env python3
"""Builds artifacts/montage.png: input (top) vs. GPU edge maps (bottom)."""
import glob
import os

from PIL import Image

ins = sorted(glob.glob("data/input/*.pgm"))[:5]
if not ins:
    raise SystemExit("no inputs")
w = 256
canvas = Image.new("L", (w * len(ins), w * 2))
for i, path in enumerate(ins):
    out = os.path.join("data/output", "edges_" + os.path.basename(path))
    canvas.paste(Image.open(path).resize((w, w)), (i * w, 0))
    canvas.paste(Image.open(out).resize((w, w)), (i * w, w))
canvas.save("artifacts/montage.png")
print("Wrote artifacts/montage.png")
