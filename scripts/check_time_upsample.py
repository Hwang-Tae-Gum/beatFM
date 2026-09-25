import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/disk1/taegum/mnt/AlignBeat_beatfm")      # MemmappedNpzFile
from mmnpz import MemmappedNpzFile  
from MusicFMExtractor import MusicFMExtractor
from spect_convert import upsample_time

npz = MemmappedNpzFile("/disk1/taegum/mnt/AlignBeat/data/audio/spectrograms/gtzan.npz")
key = next(k for k in npz.files if k.endswith("/track"))
spect = torch.from_numpy(np.asarray(npz[key], np.float32)).T[None]   # (1, 128, T_bt)

mel = upsample_time(spect)
print(key, "beat this", tuple(spect.shape), "-> 100 fps", tuple(mel.shape))

ext = MusicFMExtractor()
with torch.no_grad():
    print("whole track :", tuple(ext.forward_mel(mel).shape))
    print("15 s clip   :", tuple(ext.forward_mel(upsample_time(spect[..., :750])).shape))
