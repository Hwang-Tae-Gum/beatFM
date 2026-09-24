# BeatFM 재현

**BeatFM: Improving Beat Tracking with Pre-trained Music Foundation Model** (ICME 2025) 재현 코드입니다.
목표는 논문 Table I의 GTZAN 점수를 다시 얻는 것입니다.

| GTZAN (BeatFM, MusicFM) | F-measure | CMLt | AMLt |
|---|---|---|---|
| Beat | 89.1 | 80.6 | 93.5 |
| Downbeat | 79.6 | 74.4 | 88.7 |

**현재 상태:** 모델, 데이터 파이프라인, 학습, 평가 코드가 끝까지 연결돼 있고, CPU에서 소규모로 전체 흐름(학습 → validation → 체크포인트 → test 점수표)을 확인했습니다. 본 학습은 아직 돌리지 않았습니다.

---

## 구조

```
wav (B, samples) @ 24 kHz
  → MusicFMExtractor   frozen MusicFM, 층별 hidden state를 쌓음      → h  (B, N=13, F=1024, T)   Eq.(1)(2)
  → MSAM               temporal / frequency / channel attention      → h̃ (B, N, F, T)           Eq.(3)-(10)
  → classifier         (B, T, N·F) → Linear → ReLU → Linear(→2)      → beat, downbeat logits (B, T)
  → DBN (madmom)       beat / downbeat 시각
```
T는 25 fps (MusicFM 출력 해상도 그대로, 1 frame = 40 ms).

| 파일 | 내용 |
|---|---|
| `MusicFMExtractor.py` | frozen MusicFM. `.train()`을 불러도 eval 모드 유지 (BatchNorm 통계 보호) |
| `MSAM.py` | `MSConv`, `TemporalAggregation`, `FrequencyAggregation`, `channelAggregation`, `MSAM` |
| `model.py` | `BeatFM` = extractor → MSAM → classifier |
| `data.py` | 24 kHz 캐시, 15 s clip / 5 s overlap, ±2 frame soft label, train/val/test 분할 |
| `make_manifest.py` | 데이터셋별 오디오·라벨·fold를 모아 `manifest.csv` 생성 |
| `train.py` | Lightning 학습 + 곡 전체 test (DBN, mir_eval) |
| `scripts/` | shape 확인용 디버그 스크립트 (서버 절대경로 포함) |
| `third_party/musicfm` | MusicFM 원본 (git submodule) |

---

## 설치

```bash
git clone --recursive https://github.com/Hwang-Tae-Gum/beatFM.git
cd beatFM

conda create -n musicfm python=3.10 -y
conda activate musicfm
# S3 서버 드라이버는 CUDA 12.4까지 지원 → cu124 빌드 필요
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers<5" einops librosa pytorch_lightning mir_eval tqdm setuptools cython
pip install git+https://github.com/CPJKU/madmom      # PyPI 0.16은 최신 python에서 import 실패

# MusicFM 가중치 (MSD)
wget -P third_party/musicfm/data https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json
wget -P third_party/musicfm/data https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt
```

> **transformers 5.x에서는 MusicFM이 깨집니다** (`KeyError: 'hidden_states'` — 5.x의 `Wav2Vec2ConformerEncoder`가 hidden states를 반환하지 않음). 반드시 4.x.

---

## 실행

```bash
# 1. manifest (dataset, name, audio, annotation, fold, has_downbeats)
python make_manifest.py --out manifest.csv

# 2. smoke test
python train.py --manifest manifest.csv --cache-dir /path/to/cache \
    --limit-tracks 2 --batch-size 2 --max-epochs 1 --num-workers 0 --gpu 0

# 3. 본 학습 (fold k = 8-fold CV에서 test로 뺄 fold)
python train.py --manifest manifest.csv --cache-dir /path/to/cache --fold 0 --gpu 0
```
- 첫 실행 시 모든 오디오를 24 kHz mono float16 `.npy`로 캐시합니다 (약 15 GB).
- 끝나면 best 체크포인트(val_loss 기준)로 test를 돌리고 데이터셋별 F / CMLt / AMLt 표를 출력합니다.
- 체크포인트에는 frozen MusicFM(1.3 GB)을 빼고 저장합니다 (~80 MB). 로드 시 MusicFM 가중치 파일에서 다시 채웁니다.
- 로그: `runs/fold{k}/logs/` (CSV)

---

## 데이터

라벨과 8-fold 분할은 Beat This annotation set(`/disk1/jaehoon/dataset_store/beat_this_annotations`)을 사용합니다. 오디오 경로는 `make_manifest.py`의 `AUDIO`에서 지정합니다.

| 데이터셋 | 📄 용도 | 곡 수 (오디오 있음) | 비고 |
|---|---|---|---|
| Beatles | train | 179 | Revolution 9 오디오 없음 |
| RWC Popular | train | 100 | CD/track → RM-P### 매핑, 길이로 교차 확인 |
| **Harmonix** | train | **0** | **정렬된 오디오 필요** (아래) |
| Ballroom | 8-fold | 672 | 13곡 오디오 없음 (중복 제거본) |
| Hainsworth | 8-fold | 222 | |
| SMC | 8-fold | 217 | downbeat 라벨 없음 → downbeat loss에서 제외 |
| GTZAN | test only | 993 | 6곡 오디오 없음 |

**Harmonix:** 공식 배포는 라벨과 mel spectrogram뿐이고 오디오는 YouTube에서 받아 원본에 DTW 정렬해야 합니다. `/disk3/jaehoon/beat-tracking-dataset/_raw/harmonix/audio`의 819곡은 metadata 길이와 비교해 87%가 1초 넘게 어긋나(중앙값 +7.8 s) 정렬 전 원본으로 보입니다. Griffin-Lim 복원본(`/disk4/taegum/harmonix_griffinlim`, 912곡)은 시간은 맞지만 음질 손상이 있습니다.

---

## 논문 명시 사항 vs 직접 정한 값

**📄 논문에 명시 (Sec. III, IV) → 그대로 구현**
- frozen FM, 층별 출력 concat → `h ∈ R^{b×n×f×t}`
- MS-Conv: M = 4, dilation [1, 2, 4, 8], 병렬 → concat → MLP → sigmoid; frequency는 같은 구조, 파라미터 비공유
- channel: C-AvgPool → Conv Q/K/V → softmax(QKᵀ/√C)V → Conv → sigmoid
- `Attn = Attn_t · Attn_f · Attn_c`, `h̃ = h + Attn * h`
- BCE, beat·downbeat 동시 학습, 라벨 ±2 frame 확장 (weight 0.5, 0.25)
- 15 s clip, 5 s overlap, 같은 곡은 train/val 중 한쪽에만
- Adam, lr 3e-4, batch 16, val loss 20 epoch patience early stopping
- DBN 후처리, 곡 전체 test, mir_eval, 70 ms

**논문에 없어서 직접 정한 값 (전부 인자로 변경 가능)** — 검토 부탁드립니다

| 항목 | 현재 값 | 인자 / 위치 |
|---|---|---|
| MusicFM 체크포인트 | MSD | `MusicFMExtractor.py` |
| 사용 층 N | 13 (conv 출력 + conformer 12) — 논문은 "Encoder 1…n" | `--layers` |
| 출력 fps | 25 (업샘플 없음) → ±2 frame = ±80 ms | `data.py` |
| MS-Conv kernel / padding | 3 / 길이 유지 | `MSAM.py` |
| MS-Conv 채널 해석 | N(층)을 채널로, T 또는 F 방향 conv | `MSAM.py` |
| MS-Conv 뒤 MLP | 1×1 conv 2층, hidden = N, ReLU | `MSAM.py` |
| channel Q/K/V Conv | 층당 스칼라 → `Conv1d(1, C=16, 1)`, head 1개, 출력 `Conv1d(C, 1, 1)` | `MSAM.py` |
| classifier | N·F flatten → Linear(512) → ReLU → Linear(2) ("fully connected layers") | `--classifier mlp/linear` |
| beat/downbeat 출력 | 독립 | `model.py` |
| val 비율 | 학습 곡의 10% (곡 단위) | `--val-ratio` |
| weight decay / scheduler / augmentation | 없음 | — |
| DBN | beats_per_bar [3,4], 55–215 bpm, transition_lambda 100 | `train.py` |
| test 입력 | 곡 전체 한 번에 | `--chunk-sec` |
| 평가 시 앞부분 제외 | 5 s (mir_eval 관례) | `train.py` |
| precision | fp32 | `--precision` |

---

## 확인된 것 / 남은 것

- ✅ 실제 오디오(GTZAN 30 s) → `(1, 13, 1024, 751)`, MusicFM 가중치 보존, 학습 파라미터 6.8 M / frozen 329 M
- ✅ CPU에서 학습 → val → 체크포인트 저장·로드 → test 점수표까지 동작
- ✅ 긴 곡 chunk 추론 로직 (`--chunk-sec`)
- ⏳ GPU 학습 (torch cu124 설치 후), DBN 동작 확인 (madmom GitHub판 설치 후)
- ⏳ Harmonix 오디오
