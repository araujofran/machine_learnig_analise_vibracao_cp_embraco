from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


def crest_factor(signal: np.ndarray) -> float:
    signal_rms = rms(signal)
    if signal_rms <= 1e-12:
        return 0.0
    return float(np.max(np.abs(signal)) / signal_rms)


def dominant_frequency(signal: np.ndarray, sample_rate_hz: int) -> float:
    if len(signal) < 4:
        return 0.0

    centered = signal - np.mean(signal)
    fft_vals = np.fft.rfft(centered)
    fft_freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)

    magnitudes = np.abs(fft_vals)
    if len(magnitudes) <= 1:
        return 0.0

    magnitudes[0] = 0.0
    idx = int(np.argmax(magnitudes))
    return float(fft_freqs[idx])


def band_energy(signal: np.ndarray, sample_rate_hz: int, f_low: float, f_high: float) -> float:
    centered = signal - np.mean(signal)
    fft_vals = np.fft.rfft(centered)
    fft_freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    power = np.abs(fft_vals) ** 2

    mask = (fft_freqs >= f_low) & (fft_freqs < f_high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def temperature_slope(temp: np.ndarray) -> float:
    if len(temp) < 2:
        return 0.0
    x = np.arange(len(temp))
    coef = np.polyfit(x, temp, 1)
    return float(coef[0])


def summarize_state(window_df: pd.DataFrame) -> str:
    mode = window_df["operating_state"].mode()
    if mode.empty:
        return "unknown"
    return str(mode.iloc[0])


def build_feature_row(
    window_df: pd.DataFrame,
    sample_rate_hz: int,
    source_file: str,
) -> dict:
    radial = window_df["vibration_radial_g"].to_numpy()
    axial = window_df["vibration_axial_g"].to_numpy()
    temp = window_df["temperature_c"].to_numpy()
    current = window_df["current_a"].to_numpy()

    row = {
        "source_file": source_file,
        "asset_id": str(window_df["asset_id"].iloc[0]),
        "compressor_model": str(window_df["compressor_model"].iloc[0]),
        "fault_mode": str(window_df["fault_mode"].iloc[0]) if "fault_mode" in window_df.columns else "unknown",
        "operating_state": summarize_state(window_df),
        "start_timestamp": str(window_df["timestamp"].iloc[0]),
        "end_timestamp": str(window_df["timestamp"].iloc[-1]),
        "start_time_s": float(window_df["time_s"].iloc[0]),
        "end_time_s": float(window_df["time_s"].iloc[-1]),
        "window_size_samples": int(len(window_df)),

        "radial_rms": rms(radial),
        "axial_rms": rms(axial),
        "radial_std": float(np.std(radial)),
        "axial_std": float(np.std(axial)),
        "radial_peak": float(np.max(np.abs(radial))),
        "axial_peak": float(np.max(np.abs(axial))),
        "radial_crest_factor": crest_factor(radial),
        "axial_crest_factor": crest_factor(axial),
        "radial_kurtosis": float(kurtosis(radial, fisher=False, bias=False)),
        "axial_kurtosis": float(kurtosis(axial, fisher=False, bias=False)),
        "radial_dominant_freq_hz": dominant_frequency(radial, sample_rate_hz),
        "axial_dominant_freq_hz": dominant_frequency(axial, sample_rate_hz),

        "radial_energy_0_30hz": band_energy(radial, sample_rate_hz, 0, 30),
        "radial_energy_30_80hz": band_energy(radial, sample_rate_hz, 30, 80),
        "radial_energy_80_200hz": band_energy(radial, sample_rate_hz, 80, 200),

        "axial_energy_0_30hz": band_energy(axial, sample_rate_hz, 0, 30),
        "axial_energy_30_80hz": band_energy(axial, sample_rate_hz, 30, 80),
        "axial_energy_80_200hz": band_energy(axial, sample_rate_hz, 80, 200),

        "temp_mean": float(np.mean(temp)),
        "temp_max": float(np.max(temp)),
        "temp_min": float(np.min(temp)),
        "temp_delta": float(np.max(temp) - np.min(temp)),
        "temp_slope": temperature_slope(temp),

        "current_mean": float(np.mean(current)),
        "current_max": float(np.max(current)),
        "current_min": float(np.min(current)),
        "current_std": float(np.std(current)),
        "current_delta": float(np.max(current) - np.min(current)),
        "current_inrush_flag": int(np.max(current) > 2.0),

        "is_healthy": int(str(window_df["fault_mode"].iloc[0]) == "healthy"),
    }

    return row 
