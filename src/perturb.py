"""Test-time corruptions for RQ1 (never used in training).

Waveform-level: noise (pink, at a target SNR), gain (dB), tmask (zeroed contiguous segment).
Mel-power-level: fmask (zeroed contiguous block of Mel bands, applied before the log).

Every corruption is seeded by crc32(sample_id, type, level), so all models and seeds see identical corrupted inputs.
Severity is the index 1..3 into `robustness.levels` in configs/default.yaml.
"""
import zlib

import numpy as np
import torch

WAVEFORM_TYPES = ("noise", "gain", "tmask")
SPEC_TYPES = ("fmask",)


def rng_for(sample_id: str, kind: str, level: float) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{sample_id}|{kind}|{level}".encode()))


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance Gaussian noise with a 1/f power spectrum."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return (x / x.std()).astype(np.float32)


def audible_span(wav: np.ndarray) -> np.ndarray:
    """Part of the window that holds the clip (zero padding excluded), used to define the SNR."""
    nz = np.flatnonzero(wav)
    return wav[nz[0]:nz[-1] + 1] if len(nz) else wav


def add_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add pink noise over the whole window; SNR is measured against the clip's own span, not the zero padding."""
    p_sig = float(np.mean(audible_span(wav).astype(np.float64) ** 2))
    if p_sig == 0.0:
        return wav
    noise = pink_noise(len(wav), rng)
    noise *= np.sqrt(p_sig / 10 ** (snr_db / 10))
    return (wav + noise).astype(np.float32)


def apply_gain(wav: np.ndarray, gain_db: float) -> np.ndarray:
    """Pure scaling, no clipping (log-Mel then shifts by a constant)."""
    return (wav * 10 ** (gain_db / 20)).astype(np.float32)


def time_mask(wav: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Zero one contiguous segment covering `frac` of the window at a random position."""
    w = int(round(frac * len(wav)))
    start = int(rng.integers(0, len(wav) - w + 1))
    out = wav.copy()
    out[start:start + w] = 0.0
    return out


def make_waveform_perturb(kind: str, level: float):
    """Return perturb(wav, sample_id) for AudioClips, or None for the clean condition."""
    if kind == "noise":
        return lambda wav, sid: add_noise(wav, level, rng_for(sid, kind, level))
    if kind == "gain":
        return lambda wav, sid: apply_gain(wav, level)
    if kind == "tmask":
        return lambda wav, sid: time_mask(wav, level, rng_for(sid, kind, level))
    return None


def make_power_hook(kind: str, level: float, sample_ids: list[str]):
    """Return hook(spec) zeroing a contiguous block of `level` Mel bands per clip (spec: B, n_mels, frames)."""
    def hook(spec: torch.Tensor) -> torch.Tensor:
        spec = spec.clone()
        n_mels = spec.shape[1]
        for b, sid in enumerate(sample_ids):
            start = int(rng_for(sid, kind, level).integers(0, n_mels - int(level) + 1))
            spec[b, start:start + int(level)] = 0.0
        return spec
    return hook
