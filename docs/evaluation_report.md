# 📋 Relatório de Validação do Compressor (EM2X3125U)

Este relatório é gerado automaticamente pelo pipeline. Ele avalia o modelo preditivo construído para alarmar anomalias de vibração e corrente.

## 1️⃣ Análise Falsos Positivos em Regimes Não-Produtivos

✅ **Perfeito!** O filtro de _Uptime_ operou maravilhosamente bem. Falsos alarmes estão anulados durante o repouso e a partida.
| operating_state   |   windows |   false_positive_rate |   mean_normality_score |
|:------------------|----------:|----------------------:|-----------------------:|
| steady_state      |       255 |             0.0509804 |              0.0545661 |
| idle              |       134 |             0         |             -0.203666  |
| shutdown          |        30 |             0         |             -0.203776  |
| startup           |        30 |             0         |             -0.228743  |

## 2️⃣ Taxa de Detecção por Modo de Falha (Recall em Steady State)

Observa-se abaixo a performance do modelo em identificar cada tipo de problema induzido no maquinário quando o mesmo está em velocidade máxima.

- **MECHANICAL_LOOSENESS** - Acerto: **56.8%** das janelas. 
  > 🔴 Alerta Crítico! Falha mascarada. Você deve rever o limite de *Z-Score* híbrido e baixar o percentil.

- **MISALIGNMENT** - Acerto: **56.8%** das janelas. 
  > 🔴 Alerta Crítico! Falha mascarada. Você deve rever o limite de *Z-Score* híbrido e baixar o percentil.

- **UNBALANCE** - Acerto: **56.8%** das janelas. 
  > 🔴 Alerta Crítico! Falha mascarada. Você deve rever o limite de *Z-Score* híbrido e baixar o percentil.

- **BEARING_FAULT** - Acerto: **54.8%** das janelas. 
  > 🔴 Alerta Crítico! Falha mascarada. Você deve rever o limite de *Z-Score* híbrido e baixar o percentil.

Tabela técnica com score de normalidade puro:

| fault_mode           |   windows |   mean_normality_score |   median_normality_score |   anomaly_rate |
|:---------------------|----------:|-----------------------:|-------------------------:|---------------:|
| mechanical_looseness |       449 |             -0.203036  |               -0.199761  |      0.567929  |
| misalignment         |       449 |             -0.180187  |               -0.170796  |      0.567929  |
| unbalance            |       449 |             -0.146343  |               -0.116389  |      0.567929  |
| bearing_fault        |       449 |             -0.133739  |               -0.133397  |      0.547884  |
| healthy              |       449 |             -0.0586916 |                0.0135533 |      0.0289532 |

---
_Gerado por Plataforma Preditiva Embraco (Inteligência Nocode)_