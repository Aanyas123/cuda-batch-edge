#!/usr/bin/env python3
"""Converts a folder of PNG/JPG/TIFF images (e.g. USC-SIPI) to 8-bit PGM."""
import argparse
import os

from PIL import Image

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst", nargs="?", default="data/input")
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(args.src)):
        if name.lower().endswith(EXTS):
            img = Image.open(os.path.join(args.src, name)).convert("L")
            img.save(os.path.join(args.dst, os.path.splitext(name)[0] + ".pgm"))
            n += 1
    print(f"Converted {n} images to {args.dst}")


if __name__ == "__main__":
    main()
