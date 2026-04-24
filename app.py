import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="K-Moda · Disciplina de Inversión Comercial",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F5F0E8;
    color: #1C1A16;
}
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 300;
    color: #1C1A16;
}
h1 { font-size: 3.15rem; letter-spacing: -0.02em; }
h2 { font-size: 1.8rem; letter-spacing: -0.01em; }
h3 { font-size: 1.25rem; }

.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1450px; }

.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E0D8CC;
    border-radius: 2px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 0.6rem;
}
.kpi-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8C7B6B;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.15rem;
    line-height: 1;
    color: #1C1A16;
}
.kpi-sub {
    font-size: 0.74rem;
    color: #A09080;
    margin-top: 0.3rem;
    line-height: 1.5;
}

.section-divider {
    border: none;
    border-top: 1px solid #D4C9B8;
    margin: 2.5rem 0;
}

.section-num {
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #B0986A;
    margin-bottom: 0.2rem;
}

.insight-box {
    background: #FFFDF7;
    border-left: 3px solid #B0986A;
    padding: 1rem 1.35rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    line-height: 1.75;
    color: #3C3328;
}

.metric-strip {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.metric-pill {
    background: #F0EBE0;
    border: 1px solid #D4C9B8;
    padding: 4px 12px;
    border-radius: 2px;
    font-size: 0.72rem;
    color: #6D5E4E;
}

.badge-pos { background:#E8F5E9; color:#2E7D32; padding:2px 10px; border-radius:2px; font-size:0.75rem; font-weight:500; }
.badge-neg { background:#FBE9E7; color:#BF360C; padding:2px 10px; border-radius:2px; font-size:0.75rem; font-weight:500; }
.badge-neu { background:#F3EFE8; color:#6D5E4E; padding:2px 10px; border-radius:2px; font-size:0.75rem; font-weight:500; }

.mix-card {
    background: #FFFFFF;
    border: 1px solid #E0D8CC;
    padding: 1rem 1.2rem;
    min-height: 130px;
}

.small-note {
    font-size: 0.78rem;
    color: #8C7B6B;
    line-height: 1.6;
}
</style>
""",
    unsafe_allow_html=True,
)


DATA_PROC = os.path.join("data", "processed")
DATA_RAW = os.path.join("data", "raw")

MODEL_METRICS = {
    "mape_cv_total": 10.28,
    "mape_cv_ex_covid": 5.57,
    "r2_cv_total": -1.0518,
    "mape_holdout": 18.11,
    "r2_holdout": -0.3089,
    "mape_in": 0.86,
    "r2_in": 0.9973,
}

CANAL_A_BLOQUE = {
    "Paid Search": "bloque_digital_perf",
    "Social Paid": "bloque_digital_perf",
    "Display": "bloque_digital_awareness",
    "Video Online": "bloque_digital_awareness",
    "Exterior": "bloque_offline",
    "Prensa": "bloque_offline",
    "Radio Local": "bloque_offline",
    "Email CRM": "bloque_crm",
}

BLOQUES = [
    "bloque_digital_perf",
    "bloque_digital_awareness",
    "bloque_offline",
    "bloque_crm",
]

BLOQUES_LABEL = {
    "bloque_digital_perf": "Digital Performance",
    "bloque_digital_awareness": "Digital Awareness",
    "bloque_offline": "Offline",
    "bloque_crm": "CRM / Email",
}


def mpl_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#F5F0E8",
            "axes.facecolor": "#F5F0E8",
            "axes.edgecolor": "#D4C9B8",
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E8E0D4",
            "grid.linewidth": 0.5,
            "xtick.color": "#8C7B6B",
            "ytick.color": "#8C7B6B",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "serif",
            "text.color": "#1C1A16",
        }
    )


mpl_style()


@st.cache_data
def cargar_datos():
    df = pd.read_csv(
        os.path.join(DATA_PROC, "dataset_maestro_semanal.csv"),
        parse_dates=["fecha"],
    ).sort_values("fecha").reset_index(drop=True)

    inv = pd.read_csv(
        os.path.join(DATA_RAW, "inversion_medios_semanal.csv"),
        parse_dates=["semana_inicio", "semana_fin"],
    )

    betas = pd.read_csv(os.path.join(DATA_PROC, "betas_bloques.csv"))
    base_features = pd.read_csv(os.path.join(DATA_PROC, "base_features_mmm.csv"))["feature"].tolist()
    media_features = pd.read_csv(os.path.join(DATA_PROC, "media_features_mmm.csv"))["feature"].tolist()

    return df, inv, betas, base_features, media_features


@st.cache_resource
def cargar_modelos():
    with open(os.path.join(DATA_PROC, "modelo_mmm_base.pkl"), "rb") as f:
        model_base = pickle.load(f)
    with open(os.path.join(DATA_PROC, "scaler_mmm_base.pkl"), "rb") as f:
        scaler_base = pickle.load(f)
    with open(os.path.join(DATA_PROC, "modelo_mmm_media.pkl"), "rb") as f:
        model_media = pickle.load(f)
    with open(os.path.join(DATA_PROC, "scaler_mmm_media.pkl"), "rb") as f:
        scaler_media = pickle.load(f)
    return model_base, scaler_base, model_media, scaler_media


def preparar_features(df_in):
    df_p = df_in.copy().sort_values("fecha").reset_index(drop=True)
    df_p["bloque_digital_perf"] = df_p["ads_paid_search"] + df_p["ads_social_paid"]
    df_p["bloque_digital_awareness"] = df_p["ads_display"] + df_p["ads_video_online"]
    df_p["bloque_offline"] = df_p["ads_exterior"] + df_p["ads_prensa"] + df_p["ads_radio_local"]
    df_p["bloque_crm"] = df_p["ads_email_crm"]
    for bloque in BLOQUES:
        df_p[f"x_{bloque}"] = np.log1p(df_p[bloque].clip(lower=0))
    df_p["tendencia"] = np.arange(len(df_p))
    for yr in [2021, 2022, 2023, 2024]:
        df_p[f"dummy_{yr}"] = (df_p["fecha"].dt.year >= yr).astype(int)
    return df_p


def predecir_componentes(df_in, base_features, media_features, modelos):
    model_base, scaler_base, model_media, scaler_media = modelos
    xb = scaler_base.transform(df_in[base_features])
    xm = scaler_media.transform(df_in[media_features])
    base_pred = np.maximum(model_base.predict(xb), 0)
    media_pred = np.maximum(model_media.predict(xm), 0)
    total_pred = base_pred + media_pred
    return base_pred, media_pred, total_pred


def calcular_inversion_2024(inv_raw):
    inv_2024 = inv_raw[inv_raw["semana_inicio"].dt.year == 2024].copy()
    inv_2024["bloque"] = inv_2024["canal_medio"].map(CANAL_A_BLOQUE)
    inv_bloques = (
        inv_2024.groupby("bloque")["inversion_eur"]
        .sum()
        .reset_index()
        .sort_values("inversion_eur", ascending=False)
    )
    inv_bloques["label"] = inv_bloques["bloque"].map(BLOQUES_LABEL)
    return inv_bloques


def contribucion_contrafactual(df_2024, base_features, media_features, modelos):
    base_pred, media_pred, total_pred = predecir_componentes(
        df_2024, base_features, media_features, modelos
    )
    df_eval = df_2024.copy()
    df_eval["pred_base"] = base_pred
    df_eval["pred_media"] = media_pred
    df_eval["pred_total"] = total_pred

    rows = []
    for bloque in BLOQUES:
        df_cf = df_eval.copy()
        df_cf[bloque] = 0
        df_cf[f"x_{bloque}"] = 0
        _, _, total_cf = predecir_componentes(df_cf, base_features, media_features, modelos)
        venta_attr = float((df_eval["pred_total"].values - total_cf).sum())
        rows.append(
            {
                "bloque": bloque,
                "label": BLOQUES_LABEL[bloque],
                "venta_atribuida": venta_attr,
            }
        )
    contrib_df = pd.DataFrame(rows).sort_values("venta_atribuida", ascending=False)
    return df_eval, contrib_df


def simular_mix(df_2024, base_features, media_features, modelos, target_alloc):
    df_scn = df_2024.copy()
    inv_bloques = calcular_inversion_2024(inv_raw)
    inv_actual = dict(zip(inv_bloques["bloque"], inv_bloques["inversion_eur"]))

    for bloque, inv_obj in target_alloc.items():
        inv_now = inv_actual.get(bloque, 0.0)
        factor = inv_obj / inv_now if inv_now > 0 else 0.0
        df_scn[bloque] = df_2024[bloque] * factor
        df_scn[f"x_{bloque}"] = np.log1p(df_scn[bloque].clip(lower=0))

    _, media_pred, total_pred = predecir_componentes(df_scn, base_features, media_features, modelos)
    return float(media_pred.sum()), float(total_pred.sum())


df_raw, inv_raw, betas_df, base_features, media_features = cargar_datos()
modelos = cargar_modelos()
df = preparar_features(df_raw)
df_2024 = df[df["fecha"].dt.year == 2024].copy().reset_index(drop=True)
inv_bloques_2024 = calcular_inversion_2024(inv_raw)
betas_map = dict(zip(betas_df["bloque"], betas_df["beta_real"]))

df_eval_2024, contrib_df = contribucion_contrafactual(df_2024, base_features, media_features, modelos)

mroi_rows = []
for _, row in contrib_df.iterrows():
    bloque = row["bloque"]
    inv = float(inv_bloques_2024.loc[inv_bloques_2024["bloque"] == bloque, "inversion_eur"].sum())
    venta = float(row["venta_atribuida"])
    roas_raw = venta / inv if inv > 0 else 0.0
    roi_raw = (venta * 0.55) / inv if inv > 0 else 0.0
    mroi_rows.append(
        {
            "bloque": bloque,
            "label": BLOQUES_LABEL[bloque],
            "beta": float(betas_map.get(bloque, 0.0)),
            "inv": inv,
            "venta_attr": venta,
            "roas_raw": roas_raw,
            "roi_raw": roi_raw,
        }
    )
mroi_df = pd.DataFrame(mroi_rows).sort_values("roas_raw", ascending=False)

venta_total_2024 = float(df_2024["venta_neta_eur"].sum())
base_total_2024 = float(df_eval_2024["pred_base"].sum())
incremental_total_2024 = float(df_eval_2024["pred_media"].sum())
pct_incremental_2024 = incremental_total_2024 / venta_total_2024 * 100 if venta_total_2024 > 0 else 0

st.markdown(
    """
<div style="border-bottom:1px solid #D4C9B8; padding-bottom:1.5rem; margin-bottom:2rem;">
    <div style="font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:#B0986A;margin-bottom:0.4rem;">
        K-MODA · Marketing Mix Modeling · 2020–2024
    </div>
    <h1 style="margin:0;">Disciplina de Inversión<br><em>y Defensa de Cuota</em></h1>
    <p style="color:#6D5E4E;font-size:0.95rem;margin-top:0.8rem;max-width:760px;line-height:1.8;">
        Esta lectura del MMM está planteada para toma de decisiones: cuánto negocio parece estructural,
        cuánto depende realmente de marketing y qué nivel de prudencia financiera conviene aplicar
        cuando la sensibilidad marginal del mix es baja.
    </p>
    <div class="metric-strip">
        <span class="metric-pill">MMM en 2 etapas</span>
        <span class="metric-pill">Base estructural + capa incremental</span>
        <span class="metric-pill">log1p(adstock) para saturación</span>
        <span class="metric-pill">262 semanas</span>
        <span class="metric-pill">4 bloques estratégicos</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-num">01 · Resumen Ejecutivo</div>', unsafe_allow_html=True)
st.markdown('<h2>Qué puede concluir dirección con este modelo</h2>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "Ventas 2024", f"{venta_total_2024/1e6:.1f} M€", "Serie semanal agregada"),
    (c2, "Base estructural", f"{base_total_2024/1e6:.1f} M€", f"{base_total_2024/venta_total_2024*100:.1f}% del total"),
    (c3, "Incremental modelo", f"{incremental_total_2024/1e6:.3f} M€", f"{pct_incremental_2024:.2f}% atribuible a medios"),
    (c4, "MAPE CV total", f"{MODEL_METRICS['mape_cv_total']:.2f}%", f"Holdout 2024: {MODEL_METRICS['mape_holdout']:.2f}%"),
    (c5, "R² holdout 2024", f"{MODEL_METRICS['r2_holdout']:.2f}", "Más útil que el R² in-sample para juzgar generalización"),
]
for col, label, val, sub in kpis:
    with col:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown(
    f"""
<div class="insight-box">
    <strong>Conclusión principal:</strong> el modelo final separa una <em>base estructural</em>
    de <strong>{base_total_2024/1e6:.1f} M€</strong> y una capa incremental de marketing de solo
    <strong>{incremental_total_2024/1e6:.3f} M€</strong> en 2024.
    Eso no significa que marketing “no importe”, sino que para K-Moda la mayor parte del negocio
    parece apoyarse en marca, red comercial, recurrencia y demanda ya consolidada.
    En este contexto, marketing se interpreta mejor como herramienta de estabilidad competitiva
    y defensa de cuota que como motor aislado de crecimiento rápido.
    Por eso la lectura prioriza métricas fuera de muestra y criterio económico, no un R² in-sample espectacular.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-num">02 · Fiabilidad Del Modelo</div>', unsafe_allow_html=True)
st.markdown('<h2>Qué métricas son útiles y cuáles no</h2>', unsafe_allow_html=True)

left, right = st.columns([1.2, 1])

with left:
    diag_df = pd.DataFrame(
        [
            {"Métrica": "MAPE CV total", "Valor": f"{MODEL_METRICS['mape_cv_total']:.2f}%"},
            {"Métrica": "MAPE CV ex-COVID", "Valor": f"{MODEL_METRICS['mape_cv_ex_covid']:.2f}%"},
            {"Métrica": "MAPE holdout 2024", "Valor": f"{MODEL_METRICS['mape_holdout']:.2f}%"},
            {"Métrica": "R² CV total", "Valor": f"{MODEL_METRICS['r2_cv_total']:.4f}"},
            {"Métrica": "R² holdout 2024", "Valor": f"{MODEL_METRICS['r2_holdout']:.4f}"},
            {"Métrica": "R² in-sample (solo referencia)", "Valor": f"{MODEL_METRICS['r2_in']:.4f}"},
        ]
    )
    st.dataframe(diag_df, hide_index=True, use_container_width=True)

with right:
    st.markdown(
        """
<div class="mix-card">
    <div class="kpi-label">Lectura para comité</div>
    <div class="small-note">
        El modelo final es más limpio que la versión inicial porque elimina variables demasiado próximas a la venta.
        Aun así, el holdout 2024 sigue siendo débil. La consecuencia práctica es clara:
        no conviene presentar este MMM como prueba de una optimización agresiva,
        sino como herramienta de prudencia presupuestaria y control del mix.
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="mix-card" style="margin-top:0.8rem;">
    <div class="kpi-label">Por Qué No Vendemos El R² = 0.99</div>
    <div class="small-note">
        El <code>R² in-sample ≈ 0.99</code> queda alto porque evalúa el ajuste sobre la misma muestra usada
        para entrenar, con controles estructurales y una descomposición base + media.
        Eso no demuestra capacidad predictiva real fuera de muestra.
        Para juzgar utilidad directiva del modelo nos apoyamos sobre todo en <strong>MAPE CV</strong>
        y <strong>holdout 2024</strong>, que son bastante más exigentes.
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="insight-box">
    <strong>Traducción para CFO:</strong> un R² in-sample muy alto puede coexistir con una capacidad baja
    para apoyar decisiones de inversión. Si el holdout es débil, es más serio admitir esa limitación
    que convertir el modelo en una falsa demostración de precisión.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-num">03 · Composición Del Negocio</div>', unsafe_allow_html=True)
st.markdown('<h2>Base estructural frente a contribución incremental</h2>', unsafe_allow_html=True)

col_chart, col_text = st.columns([1.8, 1])

with col_chart:
    resumen_anual = []
    df_full = df.copy()
    for year in range(2020, 2025):
        df_y = df_full[df_full["fecha"].dt.year == year].copy().reset_index(drop=True)
        if len(df_y) == 0:
            continue
        base_y, media_y, total_y = predecir_componentes(df_y, base_features, media_features, modelos)
        resumen_anual.append(
            {
                "anio": str(year),
                "real": df_y["venta_neta_eur"].sum(),
                "base": base_y.sum(),
                "media": media_y.sum(),
                "pred": total_y.sum(),
            }
        )
    res = pd.DataFrame(resumen_anual)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(res))
    ax.bar(x, res["base"] / 1e6, color="#C8B89A", label="Base estructural", width=0.55)
    ax.bar(
        x,
        res["media"] / 1e6,
        bottom=res["base"] / 1e6,
        color="#B0986A",
        label="Incremental marketing",
        width=0.55,
    )
    ax.plot(x, res["real"] / 1e6, color="#5B4B3A", marker="o", linewidth=1.3, label="Venta real")
    ax.set_xticks(x)
    ax.set_xticklabels(res["anio"])
    ax.set_ylabel("Millones €", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M€"))
    ax.legend(fontsize=8, framealpha=0)
    ax.set_title("Base estructural + incremental frente a venta real", fontsize=10, pad=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_text:
    for _, row in res.iterrows():
        pct = row["media"] / row["real"] * 100 if row["real"] > 0 else 0
        badge = "badge-neu" if pct < 1 else "badge-pos"
        st.markdown(
            f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    border-bottom:1px solid #E8E0D4;padding:0.6rem 0;font-size:0.84rem;">
            <span style="color:#6D5E4E;font-weight:500;">{row['anio']}</span>
            <span>{row['real']/1e6:.1f} M€</span>
            <span class="{badge}">{pct:.2f}% incremental</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="insight-box">
    La señal más robusta del caso no es “qué bloque gana más”, sino que la parte incremental
    es muy pequeña frente al total. Eso cambia la conversación financiera:
    menos foco en buscar un gran uplift por reasignación y más foco en evitar sobreinversión,
    sostener presencia competitiva y exigir evidencia adicional antes de escalar gasto.
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-num">04 · Retorno Observado</div>', unsafe_allow_html=True)
st.markdown('<h2>Retorno marginal estimado por bloque</h2>', unsafe_allow_html=True)

col_cards, col_plot = st.columns([1.1, 1])

with col_cards:
    for _, row in mroi_df.iterrows():
        cents = row["roas_raw"] * 100
        badge = "badge-neu" if row["roas_raw"] < 0.01 else "badge-pos"
        st.markdown(
            f"""
        <div style="background:#FFFFFF;border:1px solid #E0D8CC;border-radius:2px;
                    padding:1rem 1.2rem;margin-bottom:0.55rem;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:1.1rem;">{row['label']}</div>
                    <div style="font-size:0.72rem;color:#A09080;margin-top:0.18rem;">
                        Inversión 2024: {row['inv']/1e6:.2f} M€ · Coef. log1p: {row['beta']:+.2f}
                    </div>
                    <div style="font-size:0.72rem;color:#A09080;margin-top:0.15rem;">
                        Venta atribuida: {row['venta_attr']:,.0f} € · ROI margen: {row['roi_raw']:.4f}x
                    </div>
                </div>
                <span class="{badge}">ROAS {row['roas_raw']:.4f}x</span>
            </div>
            <div style="margin-top:0.55rem;font-size:0.75rem;color:#6D5E4E;">
                Equivale a {cents:.2f} céntimos de venta incremental por cada euro invertido.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

with col_plot:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(mroi_df["label"], mroi_df["roas_raw"], color="#B0986A", height=0.45)
    ax.axvline(0, color="#D4C9B8", linewidth=0.8)
    ax.set_xlabel("€ de venta incremental por € invertido", fontsize=8)
    ax.set_title("ROAS observado por bloque", fontsize=10, pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.4f}x"))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown(
    """
<div class="insight-box">
    <strong>Conclusión financiera:</strong> con la última corrida del MMM no hay evidencia para defender
    un caso de retorno alto tipo “7x–20x”.
    La lectura más seria es la contraria: el marketing observado en 2024 mueve poco la predicción,
    por lo que la recomendación razonable es prudencia presupuestaria, disciplina de asignación
    y uso del MMM como herramienta de control, no como justificante de expansión agresiva.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-num">05 · Sensibilidad Del Mix</div>', unsafe_allow_html=True)
st.markdown('<h2>Qué pasa si dirección mueve el reparto presupuestario</h2>', unsafe_allow_html=True)

pool_target = float(
    inv_bloques_2024.loc[
        inv_bloques_2024["bloque"].isin(
            ["bloque_digital_awareness", "bloque_digital_perf", "bloque_offline"]
        ),
        "inversion_eur",
    ].sum()
)

actual_pool = {
    "bloque_digital_awareness": float(
        inv_bloques_2024.loc[
            inv_bloques_2024["bloque"] == "bloque_digital_awareness", "inversion_eur"
        ].sum()
    ),
    "bloque_digital_perf": float(
        inv_bloques_2024.loc[
            inv_bloques_2024["bloque"] == "bloque_digital_perf", "inversion_eur"
        ].sum()
    ),
    "bloque_offline": float(
        inv_bloques_2024.loc[inv_bloques_2024["bloque"] == "bloque_offline", "inversion_eur"].sum()
    ),
}

default_total = sum(actual_pool.values()) if sum(actual_pool.values()) > 0 else 1.0
default_aw = int(round(actual_pool["bloque_digital_awareness"] / default_total * 100))
default_pf = int(round(actual_pool["bloque_digital_perf"] / default_total * 100))
default_of = int(round(actual_pool["bloque_offline"] / default_total * 100))

st.markdown(
    """
<p style="color:#6D5E4E;font-size:0.9rem;line-height:1.8;max-width:760px;">
Los sliders no tienen que sumar 100. Funcionan como <strong>intensidades relativas</strong>:
la app normaliza automáticamente el mix final manteniendo constante el mismo pool de inversión
de Awareness + Performance + Offline observado en 2024. CRM se mantiene fijo como soporte.
</p>
""",
    unsafe_allow_html=True,
)

s1, s2, s3 = st.columns(3)
with s1:
    aw_int = st.slider("Awareness", 0, 100, max(default_aw, 1))
with s2:
    pf_int = st.slider("Performance", 0, 100, max(default_pf, 1))
with s3:
    of_int = st.slider("Offline", 0, 100, max(default_of, 1))

intensidades = np.array([aw_int, pf_int, of_int], dtype=float)
if intensidades.sum() == 0:
    intensidades = np.array([1.0, 1.0, 1.0])
shares = intensidades / intensidades.sum()

target_alloc = {
    "bloque_digital_awareness": pool_target * shares[0],
    "bloque_digital_perf": pool_target * shares[1],
    "bloque_offline": pool_target * shares[2],
    "bloque_crm": float(
        inv_bloques_2024.loc[inv_bloques_2024["bloque"] == "bloque_crm", "inversion_eur"].sum()
    ),
}

media_actual, total_actual = simular_mix(
    df_2024,
    base_features,
    media_features,
    modelos,
    {
        "bloque_digital_awareness": actual_pool["bloque_digital_awareness"],
        "bloque_digital_perf": actual_pool["bloque_digital_perf"],
        "bloque_offline": actual_pool["bloque_offline"],
        "bloque_crm": target_alloc["bloque_crm"],
    },
)
media_sim, total_sim = simular_mix(df_2024, base_features, media_features, modelos, target_alloc)
delta_media = media_sim - media_actual
delta_label = f"{delta_media/1e3:+.1f} k€" if abs(delta_media) >= 5_000 else "Cambio no material"
delta_sub = "Frente al mix actual" if abs(delta_media) >= 5_000 else "La variación estimada es prácticamente nula"

k1, k2, k3, k4 = st.columns(4)
sim_kpis = [
    (k1, "Incremental estimado", f"{media_sim/1e6:.3f} M€", "Con el mix normalizado"),
    (k2, "Δ incremental", delta_label, delta_sub),
    (
        k3,
        "Mix final",
        f"{shares[0]*100:.0f}/{shares[1]*100:.0f}/{shares[2]*100:.0f}",
        "Awareness / Performance / Offline",
    ),
    (k4, "Venta total estimada", f"{total_sim/1e6:.1f} M€", "Predicción del modelo"),
]
for col, label, val, sub in sim_kpis:
    with col:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

c_mix, c_note = st.columns([1.1, 0.9])

with c_mix:
    comp_df = pd.DataFrame(
        {
            "bloque": ["Awareness", "Performance", "Offline"],
            "actual": [
                actual_pool["bloque_digital_awareness"] / 1e6,
                actual_pool["bloque_digital_perf"] / 1e6,
                actual_pool["bloque_offline"] / 1e6,
            ],
            "simulado": [
                target_alloc["bloque_digital_awareness"] / 1e6,
                target_alloc["bloque_digital_perf"] / 1e6,
                target_alloc["bloque_offline"] / 1e6,
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(comp_df))
    w = 0.34
    ax.bar(x - w / 2, comp_df["actual"], w, color="#C8B89A", label="Actual 2024")
    ax.bar(x + w / 2, comp_df["simulado"], w, color="#8C7048", label="Simulado")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["bloque"])
    ax.set_ylabel("Millones €", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}M€"))
    ax.set_title("Mix actual frente a mix normalizado", fontsize=10, pad=10)
    ax.legend(fontsize=8, framealpha=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with c_note:
    st.markdown(
        f"""
<div class="insight-box">
    <strong>Lectura directiva del simulador:</strong> aunque cambies el reparto entre bloques,
    el incremental apenas se mueve porque la capa de medios del modelo pesa poco en la predicción total.
    En esta situación, el mensaje no es “sube el ROI reasignando”, sino <em>no esperes un impacto material
    en ventas solo por mover el mix</em>.
    <br><br>
    La variación simulada frente al mix actual es de
    <strong>{delta_media/1e3:+.1f} k€</strong>, una magnitud pequeña para el tamaño del negocio.
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-num">06 · Recomendación Ejecutiva</div>', unsafe_allow_html=True)
st.markdown('<h2>Cómo debería cerrarse este caso ante dirección financiera</h2>', unsafe_allow_html=True)

e1, e2 = st.columns(2)
with e1:
    st.markdown(
        """
<div class="mix-card">
    <div class="kpi-label">Mensaje principal</div>
    <div class="small-note">
        K-Moda no aparece como un caso donde marketing explique gran parte de la venta semanal.
        El hallazgo central es la fortaleza de la base estructural. Eso es coherente con una marca madura:
        notoriedad acumulada, recurrencia, tiendas y demanda orgánica pesan más que la activación táctica.
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="mix-card" style="margin-top:0.8rem;">
    <div class="kpi-label">Qué sí podemos afirmar</div>
    <div class="small-note">
        El MMM sí sirve para evitar sobreinterpretar atribuciones fáciles, detectar que la sensibilidad marginal es baja
        y justificar una postura conservadora: proteger cuota, no sobredimensionar inversión
        y exigir más experimentación antes de prometer retornos altos.
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )
with e2:
    st.markdown(
        """
<div class="mix-card">
    <div class="kpi-label">Qué no conviene afirmar</div>
    <div class="small-note">
        No conviene defender un “modelo casi perfecto” apoyándote en el R²≈0.99.
        Tampoco conviene vender que una reasignación concreta vaya a desbloquear un gran uplift,
        porque el propio holdout y el simulador apuntan a una sensibilidad limitada.
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="mix-card" style="margin-top:0.8rem;">
    <div class="kpi-label">Cierre ejecutivo</div>
    <div class="small-note">
        La versión más sólida del caso es esta:
        el modelo no encuentra evidencia fuerte de retorno marginal alto en 2024,
        así que la recomendación es tratar marketing como inversión de estabilidad competitiva,
        no como excusa para inflar expectativas financieras ni para justificar expansión presupuestaria automática.
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    """
<div style="display:flex;justify-content:space-between;align-items:center;
            font-size:0.72rem;color:#A09080;letter-spacing:0.06em;padding-bottom:1rem;">
    <span>K-MODA · Marketing Mix Modeling · lectura ejecutiva 2024</span>
    <span>MMM 2 etapas · Base estructural + incremental · log1p(adstock)</span>
    <span>App centrada en explicación, no en sobrepromesa de ROI</span>
</div>
""",
    unsafe_allow_html=True,
)
