import sys, torch
sys.path.append('/disk1/taegum/mnt')
from musicfm.model.musicfm_25hz import MusicFM25Hz

m = MusicFM25Hz(is_flash=False,
    stat_path='/disk1/taegum/mnt/musicfm/data/msd_stats.json',
    model_path='/disk1/taegum/mnt/musicfm/data/pretrained_msd.pt').eval()

wav = torch.randn(1, 24000 * 5)
with torch.no_grad():
    logits, h = m.get_predictions(wav)
print(len(h), h[-1].shape)
