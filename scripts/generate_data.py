#!/usr/bin/env python3
"""Generates N synthetic grayscale PGM images (shapes, gradients, noise)."""
import argparse
import os

import numpy as np


def make_image(rng, size):
    yy, xx = np.mgrid[0:size, 0:size]
    img = (rng.uniform(20, 120) + rng.uniform(0, 80) *
           (xx * rng.uniform(-1, 1) + yy * rng.uniform(-1, 1)) / size)
    for _ in range(rng.integers(6, 14)):
        val = rng.uniform(0, 255)
        if rng.random() < 0.5:
            cx, cy, r = rng.integers(0, size, 2).tolist() + [rng.integers(15, size // 4)]
            img = np.where((xx - cx) ** 2 + (yy - cy) ** 2 < r * r, val, img)
        else:
            x0, y0 = rng.integers(0, size - 20, 2)
            w, h = rng.integers(15, size // 3, 2)
            img[y0:y0 + h, x0:x0 + w] = val
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/input")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.count):
        img = make_image(rng, args.size)
        with open(os.path.join(args.out, f"img_{i:04d}.pgm"), "wb") as f:
            f.write(f"P5\n{args.size} {args.size}\n255\n".encode())
            f.write(img.tobytes())
    print(f"Wrote {args.count} images ({args.size}x{args.size}) to {args.out}")


if __name__ == "__main__":
    main()
