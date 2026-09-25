"""Beat This spectrogram -> MusicFM input, compared with MusicFM's own mel from the same GTZAN audio."""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmnpz import MemmappedNpzFile
from MusicFMExtractor import DEFAULT_STAT_PATH, MusicFMExtractor
from spect_convert import bt_to_musicfm, build_freq_map

NPZ = "/disk1/taegum/mnt/AlignBeat/data/audio/spectrograms/gtzan.npz"
AUDIO = Path("/home/taegum/mnt/labeled_data/gtzan/data")
N_CALIB, N_TEST = 10, 10

torch.set_grad_enabled(False)
ext = MusicFMExtractor()
fm = ext.musicfm
stats = json.load(open(DEFAULT_STAT_PATH))
std = stats["melspec_2048_std"]
width_bt, freq_map = build_freq_map()
npz = MemmappedNpzFile(NPZ)

names = sorted(k[:-len("/track")] for k in npz.files if k.endswith("/track"))
names = [names[i] for i in np.random.default_rng(0).permutation(len(names))[:N_CALIB + N_TEST]]


def audio_mel(name):
    """MusicFM's own input from the audio: (1, 128, T)"""
    _, genre, idx = name.split("_")                       # gtzan_blues_00000
    path = next(AUDIO.glob(f"*_{genre}.{idx}.wav"))
    y, _ = librosa.load(path, sr=24000, mono=True)
    x = fm.preprocessing(torch.from_numpy(y)[None], features=["melspec_2048"])
    return fm.normalize(x)["melspec_2048"]


def pair(name, db_offset):
    a = audio_mel(name)
    spect = torch.from_numpy(np.asarray(npz[f"{name}/track"], np.float32))[None]   # (1, T_bt, 128)
    b = bt_to_musicfm(spect, stats, width_bt, freq_map, db_offset)
    T = min(a.shape[-1], b.shape[-1])
    return a[..., :T], b[..., :T]


# 1) level offset: median dB difference on calibration tracks (should come out ~34.49)
diffs = [((a - b).median() * std).item() for a, b in (pair(n, 0.0) for n in names[:N_CALIB])]
print(f"dB offset {np.median(diffs):+.2f} (spread {np.std(diffs):.2f})")

# 2) held-out tracks: mel error and MusicFM hidden-state similarity
mae, band_mae, cos = [], [], []
for n in names[N_CALIB:]:
    a, b = pair(n, 34.49)
    err = (a - b).abs()[0] * std                          # (128, T) in dB
    mae.append(err.mean().item())
    band_mae.append(err.mean(1))
    ha, hb = ext.forward_mel(a), ext.forward_mel(b)       # (1, 13, 1024, T)
    cos.append(torch.nn.functional.cosine_similarity(ha, hb, dim=2).mean(dim=(0, 2)))

band_mae = torch.stack(band_mae).mean(0)
cos = torch.stack(cos).mean(0)
print(f"mel MAE {np.mean(mae):.2f} dB")
print("MAE by band group (low -> high):", " ".join(f"{band_mae[i:i + 16].mean():.1f}" for i in range(0, 128, 16)))
print("cosine per layer:", " ".join(f"{c:.3f}" for c in cos))
