
import os, random
import numpy as np
import cv2
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ============ CONFIG - must match the sonification run being evaluated ============
SR = 22050
N_FFT = 4096                 # = n_fft_out used for your dataset conversion
HOP = N_FFT // 2
WIN = N_FFT
TARGET_LOW, TARGET_HIGH = 1000.0, 4000.0
IMG_H, IMG_W = 513, 128      # source image resolution (unchanged I-GLA preprocessing)
IMG_ROOT = r"D:\Documents\Iquisitionis\103\vehicle-type-recognition\Dataset"
CACHE_PATH = "phasenet_mags.npz"   # delete this file if you change N_FFT or IMG_ROOT
SEED = 42
TEST_FRAC = 0.1
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
# ===================================================================================

N_BINS = N_FFT // 2 + 1
_freqs = np.linspace(0, SR / 2, N_BINS)
BAND_IDX = np.where((_freqs >= TARGET_LOW) & (_freqs <= TARGET_HIGH))[0]
BAND_LO, BAND_HI = int(BAND_IDX[0]), int(BAND_IDX[-1]) + 1   # in-band rows = [BAND_LO:BAND_HI]
N_BAND = BAND_HI - BAND_LO                                     # 558 at N_FFT=4096
PAD_H = N_BAND + (-N_BAND) % 16                                # U-Net needs a multiple of 16
AUDIO_LEN = HOP * (IMG_W - 1)


# ---- unchanged from your pipeline ----
def frequency_remap_hires(spectrogram, sr=22050, n_fft_out=4096, target_low=1000.0, target_high=4000.0):
    n_src, n_frames = spectrogram.shape
    n_bins_out = n_fft_out // 2 + 1
    freq_axis_src = np.linspace(0, sr / 2, n_src)
    b = target_low
    a = (target_high - target_low) / np.log(freq_axis_src[-1] + 1)
    f_new = a * np.log(freq_axis_src + 1) + b
    dst_pos = f_new / (sr / n_fft_out)
    boundaries = (dst_pos[:-1] + dst_pos[1:]) / 2.0
    remapped = np.zeros((n_bins_out, n_frames), dtype=spectrogram.dtype)
    freq_axis_out = np.linspace(0, sr / 2, n_bins_out)
    in_band = (freq_axis_out >= target_low) & (freq_axis_out <= target_high)
    for d in np.where(in_band)[0]:
        src_lo, src_hi = np.searchsorted(boundaries, [d - 0.5, d + 0.5])
        src_lo = min(src_lo, n_src - 1)
        src_hi = min(max(src_hi, src_lo + 1), n_src)
        remapped[d, :] = spectrogram[src_lo:src_hi, :].mean(axis=0)
    return remapped, a, b


def image_to_magnitude(image_path):
    """Exactly your I-GLA preprocessing + frequency_remap_hires, stopped right
    before Griffin-Lim. Returns only the in-band rows (everything else is 0)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or invalid path.")
    image = cv2.resize(image, (IMG_W, IMG_H))
    image_normalized = np.flipud(image.astype(np.float32) / 255.0)
    spectrogram_db = librosa.amplitude_to_db(image_normalized + 1e-7, ref=np.max)
    spectrogram_linear = librosa.db_to_amplitude(spectrogram_db)
    spectrogram_linear, _, _ = frequency_remap_hires(spectrogram_linear, sr=SR, n_fft_out=N_FFT)
    return spectrogram_linear[BAND_LO:BAND_HI].astype(np.float32)


def load_magnitudes():
    """Builds (once) and caches train/test magnitudes with a fixed-seed split."""
    if os.path.exists(CACHE_PATH):
        d = np.load(CACHE_PATH, allow_pickle=True)
        return d["train"], d["test"], list(d["test_files"])

    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(IMG_ROOT)
                   for f in fs if f.lower().endswith(IMG_EXTS))
    random.Random(SEED).shuffle(files)
    n_test = max(1, int(len(files) * TEST_FRAC))

    def stack(file_list):
        mags, kept = [], []
        for p in tqdm(file_list, desc="Building magnitudes"):
            try:
                mags.append(image_to_magnitude(p)); kept.append(p)
            except Exception as e:
                print(f"\n[skip] {p}: {e}")
        return np.stack(mags), kept

    test, test_files = stack(files[:n_test])
    train, _ = stack(files[n_test:])
    np.savez(CACHE_PATH, train=train, test=test, test_files=np.array(test_files))
    return train, test, test_files


# ---- STFT helpers (one definition shared by PhaseNet AND GLA, so SC is comparable) ----
def _win(device):
    return torch.hann_window(WIN, device=device)

def stft_mag(audio):
    return torch.stft(audio, N_FFT, HOP, WIN, _win(audio.device), center=True,
                      pad_mode="constant", return_complex=True).abs()

def band_to_full(band):
    full = torch.zeros(band.shape[0], N_BINS, band.shape[2], dtype=band.dtype, device=band.device)
    full[:, BAND_LO:BAND_HI] = band
    return full

def istft_band(mag_band, unit_phase):
    spec = band_to_full(mag_band.to(torch.complex64) * unit_phase)
    return torch.istft(spec, N_FFT, HOP, WIN, _win(spec.device), center=True, length=AUDIO_LEN)

def spectral_convergence(audio, mag_band):
    """||S - |STFT(audio)|||_F / ||S||_F over the FULL spectrum (lower = better).
    Energy leaking outside 1-4kHz counts as error too."""
    S = band_to_full(mag_band)
    S_hat = stft_mag(audio)
    return torch.linalg.norm((S - S_hat).flatten(1), dim=1) / torch.linalg.norm(S.flatten(1), dim=1)


# ---- PhaseNet: small U-Net, magnitude in -> unit phase (cos, sin) out ----
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class PhaseNet(nn.Module):
    def __init__(self, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        cin = 1
        for c in chs[:-1]:
            self.downs.append(_block(cin, c)); cin = c
        self.mid = _block(chs[-2], chs[-1])
        self.ups, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.up_blocks.append(_block(c * 2, c))
        self.head = nn.Conv2d(chs[0], 2, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, blk, s in zip(self.ups, self.up_blocks, reversed(skips)):
            x = blk(torch.cat([up(x), s], 1))
        return self.head(x)

def phase_from_model(model, mag_band):
    """(B, N_BAND, T) linear magnitude -> (B, N_BAND, T) complex unit phase."""
    x = (torch.log(mag_band + 1e-5) / 11.5 + 1.0).unsqueeze(1)   # roughly [0, 1]
    x = F.pad(x, (0, 0, 0, PAD_H - N_BAND))
    out = model(x)[:, :, :N_BAND].float()
    re, im = out[:, 0], out[:, 1]
    norm = torch.sqrt(re ** 2 + im ** 2 + 1e-8)
    return torch.complex(re / norm, im / norm)
