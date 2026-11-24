#!/usr/bin/env python3
"""Reproducible tiler (replaces the empty-path call at the end of Tile.ipynb).

    python preprocessing/tile.py dataset/images/train dataset/labels/train \
        dataset/images/train_split dataset/labels/train_split

Differences from the notebook version
  * seeded, so the tile set is reproducible
  * keeps a configurable fraction of EMPTY tiles (--keep-empty, default 0.25). With the
    current images every 640 tile already contains cells, so this only matters for
    sparser frames; it was never why the dataset has no background-only tiles
  * drops boxes that are mostly cut off by the tile edge (--min-visible, default 0.5);
    a cell is kept in a tile if >=50% of its box area is inside that tile
  * a cell can still appear in two overlapping tiles; that is fine for TRAINING but
    inflates validation counts, so evaluate on full images (see README).
"""
import argparse, glob, os
import cv2
import numpy as np


def tile_one(img_path, lbl_path, out_img, out_lbl, size, overlap, min_visible, keep_empty, rng):
    img = cv2.imread(img_path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    boxes = []
    if os.path.exists(lbl_path):
        for line in open(lbl_path):
            if line.strip():
                c, cx, cy, bw, bh = map(float, line.split())
                boxes.append((int(c), (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    step = int(size * (1 - overlap))
    xs = sorted({min(x, max(0, w - size)) for x in range(0, w, step)})
    ys = sorted({min(y, max(0, h - size)) for y in range(0, h, step)})
    n = 0
    for y0 in ys:
        for x0 in xs:
            lines = []
            for c, x1, y1, x2, y2 in boxes:
                ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + size), min(y2, y0 + size)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if (ix2 - ix1) * (iy2 - iy1) < min_visible * (x2 - x1) * (y2 - y1):
                    continue
                lines.append(f"{c} {((ix1 + ix2) / 2 - x0) / size:.6f} {((iy1 + iy2) / 2 - y0) / size:.6f} "
                             f"{(ix2 - ix1) / size:.6f} {(iy2 - iy1) / size:.6f}")
            if lines or rng.random() < keep_empty:
                name = f"{stem}_x{x0}_y{y0}"
                cv2.imwrite(os.path.join(out_img, name + ".jpg"), img[y0:y0 + size, x0:x0 + size])
                open(os.path.join(out_lbl, name + ".txt"), "w").write("\n".join(lines))
                n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src_img"); ap.add_argument("src_lbl"); ap.add_argument("out_img"); ap.add_argument("out_lbl")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--keep-empty", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_img, exist_ok=True); os.makedirs(a.out_lbl, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    total = 0
    for p in sorted(glob.glob(os.path.join(a.src_img, "*.png")) + glob.glob(os.path.join(a.src_img, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        total += tile_one(p, os.path.join(a.src_lbl, stem + ".txt"), a.out_img, a.out_lbl,
                          a.size, a.overlap, a.min_visible, a.keep_empty, rng)
    print("tiles written:", total)
