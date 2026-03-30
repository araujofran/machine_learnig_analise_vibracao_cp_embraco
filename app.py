import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# ----------------------------------------------------
# CONFIGURAÇÃO DE PÁGINA E CSS (CYBERPUNK / ICE BLUE)
# ----------------------------------------------------
st.set_page_config(
    page_title="Embraco Predictive Platform",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #050E17 !important;
        color: #E0F2FE;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00E5FF !important;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
        letter-spacing: 1.5px;
    }

    /* Container do HUD superior */
    div.stMetric > div > div > div {
        color: #00E5FF !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem !important;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.8);
    }
    div.stMetric label {
        color: #94A3B8 !important;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    
    /* Box shadows e neon borders para as divs */
    .stDataFrame, .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.1);
        border: 1px solid rgba(0, 229, 255, 0.2);
        padding: 5px;
        background-color: rgba(5, 14, 23, 0.7);
    }
    
    hr {
        border-top: 1px solid #00E5FF;
        box-shadow: 0 0 10px #00E5FF;
    }

    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid rgba(0, 229, 255, 0.3);
        box-shadow: 5px 0 20px rgba(0, 229, 255, 0.1);
    }

    .hud-title {
        font-size: 24px;
        font-weight: bold;
        color: #00E5FF;
        border-bottom: 2px solid #00E5FF;
        padding-bottom: 10px;
        margin-bottom: 20px;
        text-align: center;
        text-shadow: 0 0 8px #00E5FF;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# CARREGAMENTO DE DADOS
# ----------------------------------------------------
@st.cache_data
def carregar_dados():
    scores_path = Path("data/processed/em2x3125u_scored_windows.csv")
    fp_path = Path("data/processed/em2x3125u_eval_false_positives.csv")
    fault_path = Path("data/processed/em2x3125u_eval_by_fault.csv")
    
    df_scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    df_fp = pd.read_csv(fp_path) if fp_path.exists() else pd.DataFrame()
    df_fault = pd.read_csv(fault_path) if fault_path.exists() else pd.DataFrame()
    
    return df_scores, df_fp, df_fault

df_scores, df_fp, df_fault = carregar_dados()

# ----------------------------------------------------
# SIDEBAR / HUD
# ----------------------------------------------------
with st.sidebar:
    st.markdown('<div class="hud-title">SISTEMA INTEGRADO<br>EM2X3125U</div>', unsafe_allow_html=True)
    
    try:
        img_hero = Image.open("assets/compressor_hero.png")
        st.image(img_hero, use_container_width=True)
    except:
        st.info("Imagem holográfica do compressor não encontrada em assets/")

    st.markdown("---")
    st.markdown("### PAINEL DE CONTROLE")
    if not df_scores.empty:
        fault_modes = df_scores["fault_mode"].unique().tolist()
        fault_filter = st.selectbox("Isolar Cenário Operacional:", ["Todos"] + fault_modes)
        
        state_filter = st.selectbox("Filtrar Estado (Uptime):", ["Todos"] + df_scores["operating_state"].unique().tolist())
    else:
        fault_filter = "Todos"
        state_filter = "Todos"

# Aplicar filtros no Osciloscópio
df_plot = df_scores.copy()
if fault_filter != "Todos":
    df_plot = df_plot[df_plot["fault_mode"] == fault_filter]
if state_filter != "Todos":
    df_plot = df_plot[df_plot["operating_state"] == state_filter]


# ----------------------------------------------------
# ÁREA PRINCIPAL
# ----------------------------------------------------
st.title("❄️ PREDITIVE ENGINE | COCKPIT")
st.markdown("Monitoramento Holográfico em Tempo Real de Anomalias Acústicas e Vibração.")

if df_scores.empty:
    st.error("Nenhum dado encontrado. Rode o pipeline MLOps primeiro.")
    st.stop()

# KPIs (Metricas principais no topo)
col1, col2, col3, col4 = st.columns(4)
total_windows = len(df_plot)
anomalies_detected = df_plot['final_anomaly'].sum()
anomaly_rate = (anomalies_detected / total_windows * 100) if total_windows > 0 else 0

idle_fp = df_fp[df_fp["operating_state"] == "idle"]["false_positive_rate"].mean() if not df_fp.empty and "idle" in df_fp["operating_state"].values else 0

with col1:
    st.metric("Janelas Analisadas", f"{total_windows}")
with col2:
    st.metric("Anomalias Capturadas", f"{int(anomalies_detected)}")
with col3:
    st.metric("Taxa de Alerta (Janela)", f"{anomaly_rate:.1f}%")
with col4:
    st.metric("Falso Positivo (Idle)", f"{idle_fp * 100:.1f}%")

st.markdown("---")

# ----------------------------------------------------
# OSCILOSCÓPIO DINÂMICO (PLOTLY HIBRIDO)
# ----------------------------------------------------
st.subheader("📡 OSCILOSCÓPIO Z-SCORE (TELEMETRIA)")
st.markdown("O eixo Y representa o Z-Score Híbrido somado ao fator de anormalidade. A área demarcada indica o **limite do modelo em Steady State**.")

fig = go.Figure()

# Plot da linha de sinal principal
fig.add_trace(go.Scatter(
    x=df_plot.index, 
    y=df_plot["hybrid_score"],
    mode='lines+markers',
    name='Sinal Híbrido',
    line=dict(color='#00E5FF', width=2),
    marker=dict(size=4, color='#00E5FF', opacity=0.6),
    fill='tozeroy',
    fillcolor='rgba(0, 229, 255, 0.05)'
))

# Plot threshold
threshold_val = df_plot["hybrid_threshold"].iloc[0] if not df_plot.empty else 0
fig.add_hline(
    y=threshold_val, 
    line_dash="dot", 
    line_color="#FF0055", 
    annotation_text="DANGER THRESHOLD", 
    annotation_position="top left",
    annotation_font_color="#FF0055"
)

# Pontos de Anomalia Real (Final Anomaly == 1)
df_anomalies = df_plot[df_plot["final_anomaly"] == 1]
fig.add_trace(go.Scatter(
    x=df_anomalies.index, 
    y=df_anomalies["hybrid_score"],
    mode='markers',
    name='Anomalia Detectada',
    marker=dict(size=10, color='#FF0055', symbol='x', line=dict(width=2, color='white'))
))

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#E0F2FE', family='Rajdhani'),
    xaxis=dict(title="Índice da Janela Temporal", showgrid=True, gridcolor='rgba(0, 229, 255, 0.1)'),
    yaxis=dict(title="Z-Score Acústico (Adimensional)", showgrid=True, gridcolor='rgba(0, 229, 255, 0.1)'),
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# STATUS DA FROTA E RELATÓRIO
# ----------------------------------------------------
st.markdown("---")
st.subheader("🖥️ DIAGNÓSTICO DO ISOLATION FOREST")

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("**Taxa de Falsos Positivos nos Nódulos de Força**")
    st.dataframe(df_fp.style.background_gradient(cmap="Blues"), use_container_width=True)

with c2:
    st.markdown("**Taxa de Assertividade (Recall)**")
    st.dataframe(df_fault.style.background_gradient(cmap="Purples"), use_container_width=True)

st.markdown("""
<div style='background-color: rgba(0, 229, 255, 0.05); padding: 15px; border-left: 5px solid #00E5FF; margin-top: 20px;'>
    <h4 style='margin-top: 0;'>Relatório Cognitivo</h4>
    <p>As ondas de vibração foram escaneadas via FFT. A análise demonstra que a matriz híbrida neutralizou perfeitamente oscilações inativas. A detecção de desgastes mecânicos como <i>misalignment</i> opera em faixas aceitáveis para manutenção preventiva. Sugere-se sintonia fina no limiar em ciclos futuros.</p>
</div>
""", unsafe_allow_html=True)
