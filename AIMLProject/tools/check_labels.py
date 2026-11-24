#!/usr/bin/env python3
"""Sanity checks for the YOLO dataset. Run before every training run.

    python tools/check_labels.py                # from AIMLProject/
    python tools/check_labels.py --dataset dataset

Checks
  1. every image has a label file and vice versa
  2. class ids are valid and boxes are inside the image
  3. no two images share byte-identical label files (the bug that put another
     image's labels on val/2-8-2026_(Rylan)__t058xy1c1)
  4. "do the boxes sit on cells?": ratio of edge strength inside boxes vs outside.
     Values near 1.0 mean the boxes are on background, i.e. wrong labels.
  5. no source image appears in both train and val (tile names: <source>_x<N>_y<N>)
Exit code is 1 if any hard error is found.
"""
import argparse, collections, glob, hashlib, os, re, sys

import cv2
import numpy as np

NUM_CLASSES = 5
TEXTURE_WARN = 1.3  # inside/outside edge-strength ratio below this is suspicious


def stems(folder, exts):
    out = {}
    for e in exts:
        for p in glob.glob(os.path.join(folder, e)):
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def read_boxes(path):
    rows = []
    for ln, line in enumerate(open(path), 1):
        if line.strip():
            rows.append((ln, line.split()))
    return rows


def texture_ratio(img_path, boxes):
    im = cv2.imread(img_path, 0)
    if im is None:
        return None
    im = im.astype(np.float32)
    h, w = im.shape
    g = cv2.GaussianBlur(im, (0, 0), 1.5)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mask = np.zeros((h, w), bool)
    for _, p in boxes:
        _, cx, cy, bw, bh = map(float, p)
        x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        mask[y1:y2, x1:x2] = True
    if mask.all() or not mask.any():
        return None
    return grad[mask].mean() / grad[~mask].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="full-image splits (the *_split tile folders are checked for pairing/leakage too)")
    a = ap.parse_args()

    errors, warnings = 0, 0
    hashes = collections.defaultdict(list)
    sources = collections.defaultdict(set)

    for split in a.splits + [s + "_split" for s in a.splits]:
        imgs = stems(os.path.join(a.dataset, "images", split), ["*.png", "*.jpg", "*.jpeg"])
        lbls = stems(os.path.join(a.dataset, "labels", split), ["*.txt"])
        if not imgs and not lbls:
            continue
        for s in sorted(set(imgs) - set(lbls)):
            print(f"ERROR  [{split}] image without label file: {s}"); errors += 1
        for s in sorted(set(lbls) - set(imgs)):
            print(f"ERROR  [{split}] label file without image: {s}"); errors += 1

        counts = collections.Counter()
        for s, lp in lbls.items():
            rows = read_boxes(lp)
            for ln, p in rows:
                try:
                    c = int(p[0]); vals = list(map(float, p[1:5]))
                    assert len(p) == 5
                except Exception:
                    print(f"ERROR  [{split}] {s}:{ln} malformed line"); errors += 1; continue
                if not 0 <= c < NUM_CLASSES:
                    print(f"ERROR  [{split}] {s}:{ln} class id {c} out of range"); errors += 1
                cx, cy, bw, bh = vals
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    print(f"ERROR  [{split}] {s}:{ln} box outside image"); errors += 1
                counts[c] += 1
            if not split.endswith("_split"):
                hashes[hashlib.md5(open(lp, "rb").read()).hexdigest()].append((split, s))
                if s in imgs:
                    r = texture_ratio(imgs[s], rows)
                    if r is not None and r < TEXTURE_WARN:
                        print(f"WARN   [{split}] {s}: boxes are no busier than background "
                              f"(ratio {r:.2f}) -> labels probably belong to another image"); warnings += 1
            else:
                sources[split.replace("_split", "")].add(re.sub(r"_x\d+_y\d+$", "", s))
        print(f"[{split:11s}] images={len(imgs):4d} label files={len(lbls):4d} "
              f"boxes per class={[counts.get(i, 0) for i in range(NUM_CLASSES)]}")

    for group in hashes.values():
        if len(group) > 1:
            print(f"ERROR  identical label files for different images: {group}"); errors += 1
    both = sources.get("train", set()) & sources.get("val", set())
    for s in sorted(both):
        print(f"ERROR  source image in both train and val tiles: {s}"); errors += 1

    print(f"\n{errors} error(s), {warnings} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
