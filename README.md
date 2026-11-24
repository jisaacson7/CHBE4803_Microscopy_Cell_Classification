# Microscopy cell-state detection (YOLO)

Detects 5 cell states in bright-field images: Adhered Multiplets, Adhered Singlets, Debris,
Suspended Multiplets, Suspended Singlets. Class ids follow Label Studio's `classes.txt`
export order (0-4 as listed) and are defined in `AIMLProject/cells.yaml` and `train.py`.

## Data layout (`AIMLProject/dataset/`)
- `images/{train,val}` + `labels/{train,val}`: full 2000x1494 frames and YOLO labels
- `images/{train,val}_split` + `labels/..._split`: 640x640 tiles (20% overlap) used for training
- Frames come from different days/operators; the source frame is the part of a tile name before `_x<N>_y<N>`
- `needs_relabel/`: images pulled out of the dataset until their labels are fixed

## Workflow (run from `AIMLProject/`)
1. `python tools/check_labels.py`  - pairing, class ids, duplicate label files, "boxes sit on cells" check, train/val leakage
2. `python preprocessing/tile.py <img_dir> <lbl_dir> <out_img> <out_lbl>`  - (re)tile after changing labels
3. `python train.py`  - augmentation and hyperparameters are arguments here, not in the YAML (Ultralytics ignores them there)
4. `python tools/make_folds.py --k 4` then `python train.py --folds 0` ... `3` - cross-validation by source image

## Known caveats
- Overlapping tiles contain some cells twice and clip others at the edge, so per-tile validation counts exceed the number of real cells. Report cross-validated scores and, for final numbers, evaluate on full images with merged tile predictions.
- Only 15 labeled frames; class balance differs a lot between frames. Expect noisy metrics.
- Label Studio tasks should use date-prefixed filenames. Plain `tNNNxy1c1.png` names collide across days.
