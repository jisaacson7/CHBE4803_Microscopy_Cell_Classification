`2-8-2026_(Rylan)__t058xy1c1.png` was in the validation set with the label file of
`1-29-2026_(Chris)__t058xy1c1` (byte-identical labels, different image), so its boxes sat
on empty background. Its wrong labels and tiles were removed. Relabel it in Label Studio,
then export (YOLO), put image + labels back in dataset/{images,labels}/val, and re-tile with
preprocessing/tile.py. Tip: give Label Studio tasks date-prefixed filenames so frames with the
same tNNN number from different days cannot collide.
