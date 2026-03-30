from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_data(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fault_file = eval_dir / "em2x3125u_eval_by_fault.csv"
    state_file = eval_dir / "em2x3125u_eval_by_state.csv"
    fp_file = eval_dir / "em2x3125u_eval_false_positives.csv"

    if not fault_file.exists():
        raise FileNotFoundError(f"Arquivo não achado: {fault_file}")

    df_fault = pd.read_csv(fault_file)
    df_state = pd.read_csv(state_file)
    df_fp = pd.read_csv(fp_file)

    return df_fault, df_state, df_fp


def generate_markdown_report(
    df_fault: pd.DataFrame, 
    df_state: pd.DataFrame, 
    df_fp: pd.DataFrame
) -> str:
    md = "# 📋 Relatório de Validação do Compressor (EM2X3125U)\n\n"
    md += "Este relatório é gerado automaticamente pelo pipeline. Ele avalia o modelo preditivo construído para alarmar anomalias de vibração e corrente.\n\n"

    md += "## 1️⃣ Análise Falsos Positivos em Regimes Não-Produtivos\n\n"
    
    idle_fp = df_fp[df_fp["operating_state"] == "idle"]["false_positive_rate"].mean() if "idle" in df_fp["operating_state"].values else 0
    startup_fp = df_fp[df_fp["operating_state"] == "startup"]["false_positive_rate"].mean() if "startup" in df_fp["operating_state"].values else 0
    
    if idle_fp < 0.05 and startup_fp < 0.05:
        md += "✅ **Perfeito!** O filtro de _Uptime_ operou maravilhosamente bem. Falsos alarmes estão anulados durante o repouso e a partida.\n"
    else:
        md += f"⚠️ **Alerta:** A taxa de alarme no Idle é {idle_fp*100:.1f}% e no Startup é {startup_fp*100:.1f}%. O filtro limitador pode não estar cobrindo os limites corretamente.\n"
        
    md += pd.DataFrame(df_fp).to_markdown(index=False) + "\n\n"


    md += "## 2️⃣ Taxa de Detecção por Modo de Falha (Recall em Steady State)\n\n"
    md += "Observa-se abaixo a performance do modelo em identificar cada tipo de problema induzido no maquinário quando o mesmo está em velocidade máxima.\n\n"

    for _, row in df_fault.iterrows():
        falha = row["fault_mode"]
        if falha == "healthy":
            continue
            
        rate = float(row["anomaly_rate"])
        rate_pct = rate * 100
        
        if rate_pct >= 90:
            comentario = "🚀 Excelente cobertura! O Isolation Forest hibridizado está sendo brutal nessa falha."
        elif rate_pct >= 60:
            comentario = "🟡 Mediano. O alarme ressoou em boa parte dos dados, mas perdemos anomalias menos contundentes."
        else:
            comentario = "🔴 Alerta Crítico! Falha mascarada. Você deve rever o limite de *Z-Score* híbrido e baixar o percentil."
            
        md += f"- **{falha.upper()}** - Acerto: **{rate_pct:.1f}%** das janelas. \n  > {comentario}\n\n"

    md += "Tabela técnica com score de normalidade puro:\n\n"
    md += pd.DataFrame(df_fault).to_markdown(index=False) + "\n\n"

    md += "---\n_Gerado por Plataforma Preditiva Embraco (Inteligência Nocode)_"

    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera Markdown comentado do modelo")
    parser.add_argument("--eval_dir", type=str, default="data/processed")
    parser.add_argument("--output_md", type=str, default="docs/evaluation_report.md")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    df_fault, df_state, df_fp = load_data(eval_dir)

    markdown_content = generate_markdown_report(df_fault, df_state, df_fp)

    out_file = Path(args.output_md)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(markdown_content, encoding="utf-8")

    print(f"\n[OK] Relatório comentado gravado com sucesso em: {out_file}\n")


if __name__ == "__main__":
    main()
