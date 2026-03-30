from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def load_scored_windows(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
    return pd.read_csv(path)


def summarize_by_fault(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("fault_mode")
        .agg(
            windows=("fault_mode", "count"),
            mean_normality_score=("normality_score", "mean"),
            median_normality_score=("normality_score", "median"),
            anomaly_rate=("final_anomaly", "mean"),
        )
        .reset_index()
        .sort_values(by="mean_normality_score", ascending=True)
    )


def summarize_by_state(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["fault_mode", "operating_state"])
        .agg(
            windows=("operating_state", "count"),
            mean_normality_score=("normality_score", "mean"),
            anomaly_rate=("final_anomaly", "mean"),
        )
        .reset_index()
        .sort_values(by=["fault_mode", "mean_normality_score"], ascending=[True, True])
    )


def summarize_false_positives(df: pd.DataFrame) -> pd.DataFrame:
    healthy_df = df[df["fault_mode"] == "healthy"].copy()

    return (
        healthy_df.groupby("operating_state")
        .agg(
            windows=("operating_state", "count"),
            false_positive_rate=("final_anomaly", "mean"),
            mean_normality_score=("normality_score", "mean"),
        )
        .reset_index()
        .sort_values(by="false_positive_rate", ascending=False)
    )


def save_outputs(
    summary_fault: pd.DataFrame,
    summary_state: pd.DataFrame,
    summary_fp: pd.DataFrame,
    output_dir_str: str,
) -> None:
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fault.to_csv(output_dir / "em2x3125u_eval_by_fault.csv", index=False)
    summary_state.to_csv(output_dir / "em2x3125u_eval_by_state.csv", index=False)
    summary_fp.to_csv(output_dir / "em2x3125u_eval_false_positives.csv", index=False)

    print("[OK] Avaliação por fault salva.")
    print("[OK] Avaliação por estado salva.")
    print("[OK] Avaliação de falsos positivos salva.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia as métricas do validador")
    parser.add_argument("--scored_csv", type=str, default="data/processed/em2x3125u_scored_windows.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    df = load_scored_windows(args.scored_csv)

    summary_fault = summarize_by_fault(df)
    summary_state = summarize_by_state(df)
    summary_fp = summarize_false_positives(df)

    print("=" * 90)
    print("RESUMO POR FALHA")
    print(summary_fault.to_string(index=False))

    print("\n" + "=" * 90)
    print("RESUMO POR FALHA E ESTADO OPERACIONAL")
    print(summary_state.to_string(index=False))

    print("\n" + "=" * 90)
    print("FALSOS POSITIVOS NO HEALTHY")
    print(summary_fp.to_string(index=False))

    save_outputs(summary_fault, summary_state, summary_fp, args.output_dir)


if __name__ == "__main__":
    main() 
