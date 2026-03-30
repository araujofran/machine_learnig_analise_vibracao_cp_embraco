from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing.window_em2x3125u_timeseries import process_file
from src.ml.train_em2x3125u_anomaly_model import FEATURE_COLUMNS
from src.ml.detector_model import EmbracoAnomalyDetector


def run_inference(
    input_csv: str,
    model_path: str,
    output_csv: str,
    sample_rate: int = 100,
    window_s: float = 2.0,
    step_s: float = 1.0,
) -> None:
    
    path_in = Path(input_csv)
    if not path_in.exists():
        raise FileNotFoundError(f"[ERRO] Arquivo de inferencia não achado: {path_in}")
        
    path_model = Path(model_path)
    if not path_model.exists():
        raise FileNotFoundError(f"[ERRO] Modelo não achado: {path_model}. Treine-o primeiro.")

    print(f"-> Lendo dados brutos de {path_in} e extraindo features...")
    df_features = process_file(
        file_path=path_in,
        sample_rate_hz=sample_rate,
        window_seconds=window_s,
        step_seconds=step_s
    )

    print(f"-> Carregando modelo pipeline de {path_model}...")
    model: EmbracoAnomalyDetector = joblib.load(path_model)

    X = df_features[FEATURE_COLUMNS].copy()
    context = df_features[["start_timestamp", "end_timestamp", "operating_state"]].copy()

    print("-> Realizando predição...")
    scored_df = model.predict(X, context)

    path_out = Path(output_csv)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    
    scored_df.to_csv(path_out, index=False)
    
    total_windows = len(scored_df)
    anomalies = scored_df['final_anomaly'].sum()
    
    print("="*60)
    print("RESULTADO DA INFERÊNCIA")
    print(f"Janelas analisadas: {total_windows}")
    print(f"Anomalias detectadas (final_anomaly): {anomalies}")
    print("="*60)
    print(f"[OK] Previsões salvas em: {path_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Realiza predição em um arquivo bruto de vibração")
    parser.add_argument("--input_csv", type=str, required=True, help="O CSV bruto ex: data/raw/novo_compressor.csv")
    parser.add_argument("--model_path", type=str, default="models/em2x3125u_detector_pipeline.joblib")
    parser.add_argument("--output_csv", type=str, default="data/processed/predictions.csv")
    parser.add_argument("--sample_rate", type=int, default=100)
    parser.add_argument("--window_s", type=float, default=2.0)
    parser.add_argument("--step_s", type=float, default=1.0)
    args = parser.parse_args()

    run_inference(
        input_csv=args.input_csv,
        model_path=args.model_path,
        output_csv=args.output_csv,
        sample_rate=args.sample_rate,
        window_s=args.window_s,
        step_s=args.step_s
    )

if __name__ == "__main__":
    main()
