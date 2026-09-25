"""Same GTZAN track through both input paths: are the feature frames aligned in time?"""
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmnpz import MemmappedNpzFile
from model import BeatFM

NPZ = "/disk1/taegum/mnt/AlignBeat/data/audio/spectrograms/gtzan.npz"
AUDIO = Path("/home/taegum/mnt/labeled_data/gtzan/data")
NAMES = ["gtzan_blues_00000", "gtzan_disco_00010", "gtzan_jazz_00021", "gtzan_metal_00031",
         "gtzan_pop_00041", "gtzan_rock_00050", "gtzan_hiphop_00060"]
LAGS = range(-3, 4)

torch.set_grad_enabled(False)
model = BeatFM(input_type="spect")
npz = MemmappedNpzFile(NPZ)

res = []
for name in NAMES:
    _, genre, idx = name.split("_")
    y, _ = librosa.load(next(AUDIO.glob(f"*_{genre}.{idx}.wav")), sr=24000, mono=True)
    y = y[: len(y) // 960 * 960]                                    # whole 25 fps frames

    model.input_type = "wav"                                        # reference: original BeatFM input
    hw = model.features(torch.from_numpy(y)[None])[0]               # (N, F, T)
    model.input_type = "spect"                                      # Beat This spectrogram input
    hs = model.features(torch.from_numpy(np.asarray(npz[f"{name}/track"], np.float32))[None])[0]
    print(name, "frames wav", hw.shape[-1], "| spect", hs.shape[-1])

    T = min(hw.shape[-1], hs.shape[-1]) - 6                          # room to shift +-3
    row = []
    for lag in LAGS:                                                # wav frame t vs spect frame t + lag
        a, b = hw[..., 3:3 + T], hs[..., 3 + lag:3 + lag + T]
        row.append(torch.nn.functional.cosine_similarity(a, b, dim=1).mean(-1))   # (N,)
    res.append(torch.stack(row))

r = torch.stack(res).mean(0)                                        # (lags, N)
for i, lag in enumerate(LAGS):
    print(f"lag {lag:+d} frames ({lag * 40:+4d} ms): L0 {r[i, 0]:.3f}  L6 {r[i, 6]:.3f}  L12 {r[i, 12]:.3f}")
