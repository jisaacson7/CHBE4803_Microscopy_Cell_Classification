#!/usr/bin/env python3
"""Make k-fold splits BY SOURCE IMAGE (tiles of one image never straddle train/val).

    python tools/make_folds.py --k 4

Writes dataset/folds/fold<i>_{train,val}.txt (lists of tile paths, which Ultralytics
accepts for train/val). Train with:  python train.py --folds 0   (then 1, 2, 3)
and report mean +/- std of the k validation scores instead of one 3-4 image split.
Images listed in dataset/exclude.txt (one source-image stem per line) are skipped,
e.g. images whose labels are known to be wrong.
"""
import argparse, glob, os, random, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="dataset")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

exclude = set()
ex = os.path.join(a.dataset, "exclude.txt")
if os.path.exists(ex):
    exclude = {l.strip() for l in open(ex) if l.strip()}

by_src = defaultdict(list)
for split in ("train_split", "val_split"):
    for p in sorted(glob.glob(os.path.join(a.dataset, "images", split, "*.jpg"))):
        src = re.sub(r"_x\d+_y\d+$", "", os.path.splitext(os.path.basename(p))[0])
        if src not in exclude:
            by_src[src].append(os.path.abspath(p))

srcs = sorted(by_src)
random.Random(a.seed).shuffle(srcs)
folds = [srcs[i::a.k] for i in range(a.k)]
out = os.path.join(a.dataset, "folds")
os.makedirs(out, exist_ok=True)
for i, val_srcs in enumerate(folds):
    val = [p for s in val_srcs for p in by_src[s]]
    train = [p for j, f in enumerate(folds) if j != i for s in f for p in by_src[s]]
    open(os.path.join(out, f"fold{i}_train.txt"), "w").write("\n".join(train) + "\n")
    open(os.path.join(out, f"fold{i}_val.txt"), "w").write("\n".join(val) + "\n")
    print(f"fold{i}: {len(train)} train tiles / {len(val)} val tiles; val sources: {val_srcs}")
