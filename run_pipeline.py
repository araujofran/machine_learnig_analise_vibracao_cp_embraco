from __future__ import annotations

import subprocess
import sys


def run_command(cmd: list[str]) -> None:
    print(f"\n+++ RODANDO COMANDO: {' '.join(cmd)}")
    result = subprocess.run(cmd, stderr=sys.stderr, stdout=sys.stdout)
    if result.returncode != 0:
        print(f"\n[ERRO] Comando falhou com código {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main():
    print("="*80)
    print(" INICIANDO PIPELINE EMBRACO PREDICTIVE PLATFORM ".center(80, "="))
    print("="*80)

    # 1. Simulação
    run_command([sys.executable, "-m", "src.simulation.generate_em2x3125u_dataset", "--cycles", "15"])

    # 2. Preprocessamento / Feature Engineering
    run_command([sys.executable, "-m", "src.preprocessing.window_em2x3125u_timeseries", "--window_s", "2.0"])

    # 3. Treino (Model Packaging)
    run_command([sys.executable, "-m", "src.ml.train_em2x3125u_anomaly_model", "--n_estimators", "200"])

    # 4. Avaliação
    run_command([sys.executable, "-m", "src.evaluation.evaluate_em2x3125u_model"])

    # 4.5 Gerar Relatório Técnico Comentado
    run_command([sys.executable, "-m", "src.evaluation.generate_report"])

    # 4.6 Gerar Documento de Word (.docx) com Python-Docx
    run_command([sys.executable, "-m", "src.evaluation.generate_word_report"])

    # 5. Testar Predição Solta (Inference Script)
    run_command([
        sys.executable, "-m", "src.ml.predict_em2x3125u",
        "--input_csv", "data/raw/em2x3125u_misalignment_timeseries.csv",
        "--output_csv", "data/processed/inference_test.csv"
    ])

    print("\n" + "="*80)
    print(" PIPELINE FINALIZADO COM SUCESSO ".center(80, "="))
    print("="*80)


if __name__ == "__main__":
    main()
