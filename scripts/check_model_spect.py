import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmnpz import MemmappedNpzFile
from model import BeatFM

model = BeatFM(input_type="spect")
print("buffers:", model.width_bt.shape, model.freq_map.shape)

# 1) dummy batch of 15 s clips: 750 frames @ 50 fps -> 375 frames @ 25 fps
beat, down = model(torch.rand(2, 750, 128) * 5)
print("dummy clips :", tuple(beat.shape), tuple(down.shape))                # (2, 375) (2, 375)

# 2) a real GTZAN track, whole piece
npz = MemmappedNpzFile("/disk1/taegum/mnt/AlignBeat/data/audio/spectrograms/gtzan.npz")
spect = torch.from_numpy(np.asarray(npz["gtzan_blues_00000/track"], np.float32))[None]
with torch.no_grad():
    beat, down = model(spect)
print("gtzan track :", tuple(spect.shape), "->", tuple(beat.shape))       # (1, 1501, 128) -> (1, 750)

# 3) only MSAM + classifier are trained; conversion buffers are not saved
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable {n_train / 1e6:.1f} M |", "width_bt in state_dict:", "width_bt" in model.state_dict())
