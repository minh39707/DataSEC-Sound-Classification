"""Runtime log-Mel feature extraction (never persisted to disk)."""
import torch
import torch.nn as nn
import torchaudio


class LogMel(nn.Module):
    """waveform (B, T) -> log-Mel (B, 1, n_mels, frames), optionally globally normalized."""

    def __init__(self, sample_rate: int, n_fft: int, win_length: int, hop_length: int, n_mels: int,
                 f_min: float, f_max: float, power: float = 2.0, top_db: float | None = 80.0,
                 mean: float | None = None, std: float | None = None, **_):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            f_min=f_min, f_max=f_max, n_mels=n_mels, power=power, window_fn=torch.hann_window,
            center=True, pad_mode="reflect", norm="slaney", mel_scale="slaney",
        )
        self.top_db = top_db
        self.register_buffer("mean", torch.tensor(0.0 if mean is None else mean))
        self.register_buffer("std", torch.tensor(1.0 if std is None else std))

    def set_stats(self, mean: float, std: float) -> None:
        self.mean.fill_(mean)
        self.std.fill_(std)

    def log_mel(self, wav: torch.Tensor, power_hook=None) -> torch.Tensor:
        """Un-normalized log-Mel in dB, shape (B, n_mels, frames).

        power_hook(spec) optionally corrupts the Mel power (B, n_mels, frames) before the log (test-time only).
        """
        with torch.autocast(device_type=wav.device.type, enabled=False):
            spec = self.mel(wav.float())
            if power_hook is not None:
                spec = power_hook(spec)
            db = 10.0 * torch.log10(spec.clamp_min(1e-10))
            if self.top_db is not None:
                # per-clip floor relative to that clip's peak
                db = torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - self.top_db)
        return db

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        db = self.log_mel(wav)
        return ((db - self.mean) / self.std).unsqueeze(1)


def build_logmel(cfg: dict) -> LogMel:
    return LogMel(sample_rate=cfg["audio"]["sample_rate"], **cfg["features"])
