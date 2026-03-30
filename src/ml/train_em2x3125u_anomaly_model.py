from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from src.ml.detector_model import EmbracoAnomalyDetector


FEATURE_COLUMNS = [
    "radial_rms",
    "axial_rms",
    "radial_std",
    "axial_std",
    "radial_peak",
    "axial_peak",
    "radial_crest_factor",
    "axial_crest_factor",
    "radial_kurtosis",
    "axial_kurtosis",
    "radial_dominant_freq_hz",
    "axial_dominant_freq_hz",
    "radial_energy_0_30hz",
    "radial_energy_30_80hz",
    "radial_energy_80_200hz",
    "axial_energy_0_30hz",
    "axial_energy_30_80hz",
    "axial_energy_80_200hz",
    "temp_mean",
    "temp_max",
    "temp_min",
    "temp_delta",
    "temp_slope",
    "current_mean",
    "current_max",
    "current_min",
    "current_std",
    "current_delta",
    "current_inrush_flag",
]


def load_feature_dataset(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    return pd.read_csv(path)


def split_train_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[
        (df["fault_mode"] == "healthy")
        & (df["operating_state"] == "steady_state")
    ].copy()

    eval_df = df.copy()

    if train_df.empty:
        raise ValueError("Nenhuma janela healthy em steady_state encontrada para treino.")

    return train_df, eval_df


def train_and_score(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    n_estimators: int = 400,
    contamination: str = "auto"
) -> tuple[EmbracoAnomalyDetector, pd.DataFrame]:
    
    detector = EmbracoAnomalyDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        target_operating_state="steady_state",
    )

    X_train = train_df[FEATURE_COLUMNS].copy()
    context_train = train_df[["fault_mode", "operating_state"]]
    
    detector.fit(X_train, context_train)

    X_eval = eval_df[FEATURE_COLUMNS].copy()
    context_eval = eval_df[["fault_mode", "operating_state"]]
    
    scored_df = detector.predict(X_eval, context_eval)
    
    # Ground truth
    scored_df["expected_anomaly"] = (scored_df["fault_mode"] != "healthy").astype(int)

    return detector, scored_df


def summarize_by_fault(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("fault_mode")
        .agg(
            windows=("fault_mode", "count"),
            mean_normality_score=("normality_score", "mean"),
            mean_z_score=("z_score_mean", "mean"),
            mean_hybrid_score=("hybrid_score", "mean"),
            anomaly_rate=("final_anomaly", "mean"),
        )
        .reset_index()
        .sort_values(by="mean_hybrid_score", ascending=False)
    )
    return summary


def save_outputs(
    model: EmbracoAnomalyDetector,
    scored_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    model_dir: str,
    processed_dir: str,
) -> None:
    m_dir = Path(model_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    p_dir = Path(processed_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    model_path = m_dir / "em2x3125u_detector_pipeline.joblib"
    scored_path = p_dir / "em2x3125u_scored_windows.csv"
    summary_path = p_dir / "em2x3125u_anomaly_summary.csv"

    joblib.dump(model, model_path)
    scored_df.to_csv(scored_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Modelo Pipeline salvo em: {model_path}")
    print(f"[OK] Janelas pontuadas salvas em: {scored_path}")
    print(f"[OK] Resumo salvo em: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelo Anomaly Detector do Compressor")
    parser.add_argument("--input_csv", type=str, default="data/processed/em2x3125u_all_features.csv", help="CSV de features gerado no step anterior")
    parser.add_argument("--model_dir", type=str, default="models", help="Pasta para o .joblib")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Dir save de scores")
    parser.add_argument("--n_estimators", type=int, default=400, help="Árvores do IsolationForest")
    args = parser.parse_args()

    df = load_feature_dataset(args.input_csv)
    train_df, eval_df = split_train_eval(df)

    print("=" * 90)
    print("TREINO DO DETECTOR DE ANOMALIA - EM2X3125U")
    print(f"Total de janelas: {len(df)}")
    print(f"Janelas healthy para treino: {len(train_df)}")
    print("=" * 90)

    model, scored_df = train_and_score(
        train_df=train_df,
        eval_df=eval_df,
        n_estimators=args.n_estimators
    )

    summary_df = summarize_by_fault(scored_df)

    print("\nResumo por fault_mode:")
    print(summary_df.to_string(index=False))

    print("\nClassification report (healthy vs fault) usando final_anomaly:")
    print(
        classification_report(
            scored_df["expected_anomaly"],
            scored_df["final_anomaly"],
            digits=4,
            zero_division=0,
        )
    )

    save_outputs(
        model=model,
        scored_df=scored_df,
        summary_df=summary_df,
        model_dir=args.model_dir,
        processed_dir=args.processed_dir,
    )


if __name__ == "__main__":
    main()