#!/usr/bin/env python3
"""Train the cell-state detector. Run from anywhere:

    python train.py                       # defaults below
    python train.py --imgsz 1280 --batch 4 --name big
    python train.py --folds 0             # train on fold 0 (see tools/make_folds.py)
    python train.py --dry-run             # print the resolved settings and exit

Why this exists: Ultralytics ignores augmentation keys (hsv_*, degrees, flips,
mosaic) placed in the dataset YAML. They only take effect as train() arguments,
so they live here, in version control, next to the run that used them.
"""
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
CLASS_NAMES = ["Adhered Multiplets", "Adhered Singlets", "Debris",
               "Suspended Multiplets", "Suspended Singlets"]  # == Label Studio classes.txt order


def write_runtime_yaml(train, val):
    """Write a YAML with absolute paths for this machine (cells.yaml stays portable)."""
    out = HERE / "cells.runtime.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    out.write_text(f"path: {DATASET}\ntrain: {train}\nval: {val}\nnames:\n{names}\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30, help="val plateaued ~epoch 30-60 in earlier runs")
    ap.add_argument("--imgsz", type=int, default=640, help="tiles are 640x640; 1280 only upsamples them")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="train")
    ap.add_argument("--folds", type=int, default=None,
                    help="use dataset/folds/fold<N>_{train,val}.txt from tools/make_folds.py")
    # augmentation (brightfield: no colour, orientation-free, boxes must stay tight)
    ap.add_argument("--hsv-v", type=float, default=0.3)
    ap.add_argument("--degrees", type=float, default=0.0,
                    help="rotation loosens axis-aligned boxes around ~55px cells; keep small")
    ap.add_argument("--flipud", type=float, default=0.5)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.folds is None:
        train, val = "images/train_split", "images/val_split"
    else:
        train, val = f"folds/fold{a.folds}_train.txt", f"folds/fold{a.folds}_val.txt"

    cfg = dict(
        epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch, seed=a.seed,
        cos_lr=True, close_mosaic=10, deterministic=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=a.hsv_v,
        degrees=a.degrees, flipud=a.flipud, fliplr=a.fliplr, mosaic=a.mosaic,
        project=str(HERE / "runs" / "detect"), name=a.name,
    )
    data_yaml = write_runtime_yaml(train, val)
    print("model:", a.model, "| data:", data_yaml)
    print(json.dumps(cfg, indent=2))
    if a.dry_run:
        return

    # Fail early on bad labels (duplicate/misplaced label files, class ids, leakage).
    rc = os.system(f'"{sys.executable}" "{HERE / "tools" / "check_labels.py"}" --dataset "{DATASET}"')
    if rc != 0:
        sys.exit("check_labels.py reported errors - fix the dataset first (or comment this check out).")

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=str(data_yaml), **cfg)


if __name__ == "__main__":
    main()
