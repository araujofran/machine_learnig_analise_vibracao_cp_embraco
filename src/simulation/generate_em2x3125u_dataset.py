from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd


RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


@dataclass
class CompressorProfile:
    asset_id: str
    compressor_model: str
    voltage_hz_label: str
    current_nominal_a: float
    lra_a: float
    rated_power_w: float
    rpm_ref: float


def generate_state_block(
    profile: CompressorProfile,
    start_timestamp: pd.Timestamp,
    start_time_s: float,
    duration_s: float,
    sample_rate_hz: int,
    state: str,
    base_temperature_c: float,
    steady_temperature_target_c: float,
    anomaly_mode: str = "healthy",
) -> pd.DataFrame:
    n = int(duration_s * sample_rate_hz)
    dt = 1.0 / sample_rate_hz

    time_s = start_time_s + np.arange(n) * dt
    timestamps = start_timestamp + pd.to_timedelta(np.arange(n) * dt, unit="s")

    rot_freq_hz = profile.rpm_ref / 60.0
    phase = 2 * math.pi * rot_freq_hz * (time_s - start_time_s)

    # ---------------------------
    # Corrente
    # ---------------------------
    if state == "startup":
        current = np.linspace(profile.lra_a * 0.85, profile.current_nominal_a * 1.25, n)
        current += rng.normal(0.0, 0.08, n)

    elif state == "steady_state":
        current = np.full(n, profile.current_nominal_a)
        current += rng.normal(0.0, 0.02, n)

    elif state == "shutdown":
        current = np.linspace(profile.current_nominal_a * 0.8, 0.0, n)
        current += rng.normal(0.0, 0.01, n)

    else:  # idle
        current = np.zeros(n)
        current += rng.normal(0.0, 0.002, n)

    current = np.clip(current, 0.0, None)

    # ---------------------------
    # Temperatura
    # ---------------------------
    if state == "startup":
        temperature = np.linspace(base_temperature_c, base_temperature_c + 4.0, n)
        temperature += rng.normal(0.0, 0.08, n)

    elif state == "steady_state":
        temperature = np.linspace(base_temperature_c, steady_temperature_target_c, n)
        temperature += rng.normal(0.0, 0.10, n)

    elif state == "shutdown":
        temperature = np.linspace(base_temperature_c, max(base_temperature_c - 2.0, 20.0), n)
        temperature += rng.normal(0.0, 0.06, n)

    else:  # idle
        temperature = np.linspace(base_temperature_c, max(base_temperature_c - 3.0, 20.0), n)
        temperature += rng.normal(0.0, 0.05, n)

    # ---------------------------
    # Vibração - baseline saudável
    # ---------------------------
    if state == "startup":
        radial = (
            0.10 * np.sin(phase)
            + 0.05 * np.sin(2 * phase)
            + rng.normal(0.0, 0.025, n)
        )
        axial = (
            0.07 * np.sin(phase + 0.35)
            + 0.03 * np.sin(2 * phase)
            + rng.normal(0.0, 0.020, n)
        )

    elif state == "steady_state":
        radial = (
            0.045 * np.sin(phase)
            + 0.012 * np.sin(2 * phase)
            + rng.normal(0.0, 0.006, n)
        )
        axial = (
            0.028 * np.sin(phase + 0.25)
            + 0.008 * np.sin(2 * phase)
            + rng.normal(0.0, 0.004, n)
        )

    elif state == "shutdown":
        decay = np.linspace(1.0, 0.2, n)
        radial = decay * (0.030 * np.sin(phase)) + rng.normal(0.0, 0.005, n)
        axial = decay * (0.018 * np.sin(phase + 0.20)) + rng.normal(0.0, 0.004, n)

    else:  # idle
        radial = rng.normal(0.0, 0.0015, n)
        axial = rng.normal(0.0, 0.0012, n)

    # ---------------------------
    # Anomalias em steady_state
    # ---------------------------
    if state == "steady_state" and anomaly_mode != "healthy":
        if anomaly_mode == "unbalance":
            radial += 0.080 * np.sin(phase)
            current += 0.10
            temperature += np.linspace(0.0, 1.5, n)

        elif anomaly_mode == "misalignment":
            axial += 0.060 * np.sin(2 * phase) + 0.040 * np.sin(3 * phase)
            radial += 0.020 * np.sin(phase)
            current += 0.08
            temperature += np.linspace(0.0, 1.8, n)

        elif anomaly_mode == "mechanical_looseness":
            mod = 1.0 + 0.35 * np.sin(2 * math.pi * 1.2 * (time_s - start_time_s))
            radial += mod * (0.050 * np.sin(0.5 * phase) + 0.045 * np.sin(2 * phase))
            axial += mod * 0.020 * np.sin(phase)
            current += rng.normal(0.05, 0.03, n)
            temperature += np.linspace(0.0, 1.2, n)

        elif anomaly_mode == "bearing_fault":
            hf = 0.020 * np.sin(2 * math.pi * 800 * (time_s - start_time_s))
            impulses = np.zeros(n)
            pulse_count = max(4, n // 300)
            pulse_positions = rng.choice(np.arange(20, n - 20), size=pulse_count, replace=False)
            impulses[pulse_positions] = rng.uniform(0.10, 0.22, size=pulse_count)
            kernel = np.exp(-np.linspace(0, 1, 20) * 12)
            impulse_response = np.convolve(impulses, kernel, mode="same")

            radial += hf + impulse_response
            axial += 0.5 * hf
            current += 0.06
            temperature += np.linspace(0.0, 2.5, n)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": profile.asset_id,
            "compressor_model": profile.compressor_model,
            "operating_state": state,
            "time_s": time_s,
            "vibration_radial_g": radial,
            "vibration_axial_g": axial,
            "temperature_c": temperature,
            "current_a": current,
        }
    )

    return df


def build_cycle(
    profile: CompressorProfile,
    cycle_index: int,
    start_timestamp: pd.Timestamp,
    start_time_s: float,
    sample_rate_hz: int,
    anomaly_mode: str = "healthy",
) -> pd.DataFrame:
    blocks = []

    startup_duration = 2.0
    steady_duration = 18.0
    shutdown_duration = 2.0
    idle_duration = 8.0

    base_temp = 24.0 + 0.3 * cycle_index
    steady_target = 39.0 + 0.6 * cycle_index

    t0 = start_time_s
    ts0 = start_timestamp

    startup_df = generate_state_block(
        profile=profile,
        start_timestamp=ts0,
        start_time_s=t0,
        duration_s=startup_duration,
        sample_rate_hz=sample_rate_hz,
        state="startup",
        base_temperature_c=base_temp,
        steady_temperature_target_c=steady_target,
        anomaly_mode="healthy",
    )
    blocks.append(startup_df)

    t1 = float(startup_df["time_s"].iloc[-1]) + 1.0 / sample_rate_hz
    ts1 = startup_df["timestamp"].iloc[-1] + pd.to_timedelta(1.0 / sample_rate_hz, unit="s")

    steady_df = generate_state_block(
        profile=profile,
        start_timestamp=ts1,
        start_time_s=t1,
        duration_s=steady_duration,
        sample_rate_hz=sample_rate_hz,
        state="steady_state",
        base_temperature_c=float(startup_df["temperature_c"].iloc[-1]),
        steady_temperature_target_c=steady_target,
        anomaly_mode=anomaly_mode,
    )
    blocks.append(steady_df)

    t2 = float(steady_df["time_s"].iloc[-1]) + 1.0 / sample_rate_hz
    ts2 = steady_df["timestamp"].iloc[-1] + pd.to_timedelta(1.0 / sample_rate_hz, unit="s")

    shutdown_df = generate_state_block(
        profile=profile,
        start_timestamp=ts2,
        start_time_s=t2,
        duration_s=shutdown_duration,
        sample_rate_hz=sample_rate_hz,
        state="shutdown",
        base_temperature_c=float(steady_df["temperature_c"].iloc[-1]),
        steady_temperature_target_c=steady_target,
        anomaly_mode="healthy",
    )
    blocks.append(shutdown_df)

    t3 = float(shutdown_df["time_s"].iloc[-1]) + 1.0 / sample_rate_hz
    ts3 = shutdown_df["timestamp"].iloc[-1] + pd.to_timedelta(1.0 / sample_rate_hz, unit="s")

    idle_df = generate_state_block(
        profile=profile,
        start_timestamp=ts3,
        start_time_s=t3,
        duration_s=idle_duration,
        sample_rate_hz=sample_rate_hz,
        state="idle",
        base_temperature_c=float(shutdown_df["temperature_c"].iloc[-1]),
        steady_temperature_target_c=steady_target,
        anomaly_mode="healthy",
    )
    blocks.append(idle_df)

    return pd.concat(blocks, ignore_index=True)


def generate_dataset(
    profile: CompressorProfile,
    cycles: int = 20,
    sample_rate_hz: int = 100,
    anomaly_mode: str = "healthy",
) -> pd.DataFrame:
    all_cycles = []

    start_timestamp = pd.Timestamp("2026-01-01 00:00:00")
    time_cursor = 0.0
    timestamp_cursor = start_timestamp

    for cycle_idx in range(cycles):
        cycle_df = build_cycle(
            profile=profile,
            cycle_index=cycle_idx,
            start_timestamp=timestamp_cursor,
            start_time_s=time_cursor,
            sample_rate_hz=sample_rate_hz,
            anomaly_mode=anomaly_mode,
        )
        all_cycles.append(cycle_df)

        time_cursor = float(cycle_df["time_s"].iloc[-1]) + 1.0 / sample_rate_hz
        timestamp_cursor = cycle_df["timestamp"].iloc[-1] + pd.to_timedelta(1.0 / sample_rate_hz, unit="s")

    df = pd.concat(all_cycles, ignore_index=True)
    df["fault_mode"] = anomaly_mode

    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset salvo em: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula ciclo de vida do Compressor Embraco")
    parser.add_argument("--out_dir", type=str, default="data/raw", help="Pasta para salvar")
    parser.add_argument("--cycles", type=int, default=20, help="Ciclos gerados")
    parser.add_argument("--sample_rate", type=int, default=100, help="Frequência em Hz")
    args = parser.parse_args()

    profile = CompressorProfile(
        asset_id="EMB_COMP_001",
        compressor_model="EM2X3125U",
        voltage_hz_label="220-240 V 50-60 Hz",
        current_nominal_a=0.7,
        lra_a=7.5,
        rated_power_w=182.0,
        rpm_ref=3600.0,
    )

    output_dir = Path(args.out_dir)

    healthy_df = generate_dataset(
        profile=profile,
        cycles=args.cycles,
        sample_rate_hz=args.sample_rate,
        anomaly_mode="healthy",
    )
    save_dataset(healthy_df, output_dir / "em2x3125u_healthy_timeseries.csv")

    for fault_mode in [
        "unbalance",
        "misalignment",
        "mechanical_looseness",
        "bearing_fault",
    ]:
        fault_df = generate_dataset(
            profile=profile,
            cycles=args.cycles,
            sample_rate_hz=args.sample_rate,
            anomaly_mode=fault_mode,
        )
        save_dataset(fault_df, output_dir / f"em2x3125u_{fault_mode}_timeseries.csv")


if __name__ == "__main__":
    main() 
