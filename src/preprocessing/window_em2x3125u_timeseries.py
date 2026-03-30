from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from src.features.em2x3125u_feature_builder import build_feature_row

# embraco_predictive_platform\src\features\em2x3125u_feature_builder.py


def create_windows(df: pd.DataFrame, window_size: int, step_size: int) -> list[pd.DataFrame]:
    windows = []
    n = len(df)

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        windows.append(df.iloc[start:end].copy())

    return windows


def process_file(
    file_path: Path,
    sample_rate_hz: int,
    window_seconds: float,
    step_seconds: float,
) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    window_size = int(sample_rate_hz * window_seconds)
    step_size = int(sample_rate_hz * step_seconds)

    windows = create_windows(df, window_size=window_size, step_size=step_size)

    rows = []
    for window_df in windows:
        rows.append(
            build_feature_row(
                window_df=window_df,
                sample_rate_hz=sample_rate_hz,
                source_file=file_path.name,
            )
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai e janela features do compressor")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Diretório dos CSVs")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Dir save")
    parser.add_argument("--sample_rate", type=int, default=100, help="Hz")
    parser.add_argument("--window_s", type=float, default=2.0, help="Janela")
    parser.add_argument("--step_s", type=float, default=1.0, help="Sobreposição")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("em2x3125u_*_timeseries.csv"))

    if not csv_files:
        print(f"[ERRO] Nenhum arquivo bruto encontrado em {input_dir}")
        return

    all_features = []

    for file_path in csv_files:
        df_features = process_file(
            file_path=file_path,
            sample_rate_hz=args.sample_rate,
            window_seconds=args.window_s,
            step_seconds=args.step_s,
        )

        output_file = output_dir / f"{file_path.stem}_features.csv"
        df_features.to_csv(output_file, index=False)
        print(f"[OK] Features salvas em: {output_file}")

        all_features.append(df_features)

    combined_df = pd.concat(all_features, ignore_index=True)
    combined_output = output_dir / "em2x3125u_all_features.csv"
    combined_df.to_csv(combined_output, index=False)
    print(f"[OK] Dataset consolidado salvo em: {combined_output}")


if __name__ == "__main__":
    main() 
