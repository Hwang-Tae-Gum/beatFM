import collections
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import FPS, ClipDataset, PieceDataset, read_manifest, split_rows
from model import BeatFM

rows = read_manifest("manifest_spect.csv")
train, val, test = split_rows(rows, fold=0)
for split, rs in (("train", train), ("val", val), ("test", test)):
    print(split, len(rs), dict(collections.Counter(r["dataset"] for r in rs)))

# 1) clips: 750 spect frames -> 375 label frames
ds = ClipDataset(train, input_type="spect")
item = ds[0]
print("clips", len(ds), {k: tuple(v.shape) for k, v in item.items()})

# 2) labels: frames with target 1.0 == round((beat_time - clip_start) * 25)
k = 5
i, start = ds.index[k]
beats = ds.ann[i][0]
expected = np.round((beats - start / 50) * FPS).astype(int)
expected = expected[(expected >= 0) & (expected < 375)]
print("label frames match:", np.array_equal(np.flatnonzero(ds[k]["beat"].numpy() == 1.0), expected))

# 3) a batch through the model
batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)))
with torch.no_grad():
    beat, down = BeatFM(input_type="spect")(batch["x"])
print("batch", tuple(batch["x"].shape), "-> logits", tuple(beat.shape), "| target", tuple(batch["beat"].shape))

# 4) whole piece for testing
piece = PieceDataset(test, input_type="spect")[0]
print("piece", piece["dataset"], tuple(piece["x"].shape))
