import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Análisis Admisión 2026",
    page_icon="📊",
    layout="wide"
)

st.markdown("<style>.block-container{padding-top:1.5rem}</style>", unsafe_allow_html=True)
st.title("📊 Análisis de Puntajes — Admisión 2026")
st.markdown("---")

# ── Columnas de pruebas ──────────────────────────────────────
COLS_MAP = {
    "Comprensión Lectora": "CLEC_REG_ACTUAL",
    "Matemática 1":        "MATE1_REG_ACTUAL",
    "Matemática 2":        "MATE2_REG_ACTUAL",
    "Historia y Cs. Soc.": "HCSOC_REG_ACTUAL",
    "Ciencias":            "CIEN_REG_ACTUAL",
}
LABEL_MAP = {v: k for k, v in COLS_MAP.items()}

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    sep = st.selectbox("Separador del CSV", [";", ",", "|"], index=0)
    rbd_input = st.number_input("RBD a analizar (Tab 1)", min_value=0, value=8998, step=1)
    palette_option = st.selectbox(
        "Paleta de colores",
        ["viridis", "magma", "plasma", "coolwarm", "Set2", "tab10"],
    )
    show_points = st.checkbox("Mostrar puntos individuales (stripplot)", value=False)
    st.markdown("---")
    st.caption("📌 Análisis de admisión universitaria Chile")

# ── Sin archivo ───────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Carga un archivo CSV desde el panel lateral para comenzar.")
    st.markdown("""
    ### ¿Cómo usar esta app?
    1. Sube el archivo `ArchivoC_Adm2026REG.csv` desde el panel lateral
    2. Ingresa el **RBD** del establecimiento que quieres analizar en la pestaña de distribución
    3. En la pestaña **Ranking Top 100 RBD** puedes comparar establecimientos por prueba

    **Columnas esperadas:**

    | Columna | Descripción |
    |---|---|
    | `RBD` | Código del establecimiento |
    | `CLEC_REG_ACTUAL` | Comprensión Lectora |
    | `MATE1_REG_ACTUAL` | Matemática 1 |
    | `MATE2_REG_ACTUAL` | Matemática 2 |
    | `HCSOC_REG_ACTUAL` | Historia y Cs. Sociales |
    | `CIEN_REG_ACTUAL` | Ciencias |
    """)
    st.stop()

# ── Cargar datos ─────────────────────────────────────────────
try:
    df_2026 = pd.read_csv(uploaded_file, sep=sep, low_memory=False)
except Exception as e:
    st.error(f"❌ Error al leer el archivo: {e}")
    st.stop()

if "RBD" not in df_2026.columns:
    st.error("❌ Columna 'RBD' no encontrada. Verifica el separador del CSV.")
    st.stop()

cols_present = {k: v for k, v in COLS_MAP.items() if v in df_2026.columns}
if not cols_present:
    st.error("❌ No se encontraron columnas de puntaje esperadas en el archivo.")
    st.stop()

# Selección de pruebas (sidebar)
with st.sidebar:
    st.markdown("**Pruebas a visualizar:**")
    selected_labels = st.multiselect(
        "Selecciona pruebas",
        options=list(cols_present.keys()),
        default=list(cols_present.keys())
    )

if not selected_labels:
    st.warning("Selecciona al menos una prueba en el panel lateral.")
    st.stop()

selected_cols = [cols_present[l] for l in selected_labels]

# ── Datos del RBD seleccionado ────────────────────────────────
df_rbd = df_2026[df_2026["RBD"] == rbd_input]
if df_rbd.empty:
    st.warning(f"⚠️ No se encontraron registros para RBD = {rbd_input}.")

df_filtered = df_rbd[selected_cols].copy().apply(lambda x: x.where(x != 0)) if not df_rbd.empty else pd.DataFrame(columns=selected_cols)

# ── Métricas resumen ─────────────────────────────────────────
if not df_rbd.empty:
    st.subheader(f"📋 Resumen — RBD {int(rbd_input)}")
    st.caption(f"Total de estudiantes en el establecimiento: **{len(df_rbd)}**")
    mcols = st.columns(len(selected_labels))
    for i, (label, col) in enumerate(zip(selected_labels, selected_cols)):
        serie = df_filtered[col].dropna()
        mcols[i].metric(label=label,
                        value=f"{serie.mean():.1f}" if len(serie) > 0 else "—",
                        delta=f"n = {len(serie)}")
    st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📦 Distribución de Puntajes",
    "🏆 Ranking Top 100 RBD",
    "🔍 Datos Crudos",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — Boxplot del RBD seleccionado
# ─────────────────────────────────────────────────────────────
with tab1:
    if df_rbd.empty:
        st.warning(f"No hay datos para RBD {int(rbd_input)}.")
    else:
        st.subheader(f"Distribución de Puntajes — RBD {int(rbd_input)} (ceros excluidos)")

        melted = df_filtered.melt(var_name="Test", value_name="Puntaje").dropna()
        melted["Prueba"] = melted["Test"].map(LABEL_MAP)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=melted, x="Prueba", y="Puntaje",
                    hue="Prueba", palette=palette_option, legend=False, ax=ax)
        if show_points:
            sns.stripplot(data=melted, x="Prueba", y="Puntaje",
                          color="black", alpha=0.3, size=3, jitter=True, ax=ax)

        counts = df_filtered.count().to_dict()
        ax.set_xticks(range(len(selected_cols)))
        ax.set_xticklabels(
            [f"{LABEL_MAP.get(c, c)}\n(n={counts[c]})" for c in selected_cols],
            rotation=30, ha="right"
        )
        ax.set_title(f"Distribución de Puntajes — RBD {int(rbd_input)}", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Puntaje")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📊 Estadísticas Descriptivas")
        stats = df_filtered.describe().T
        stats.index = [LABEL_MAP.get(i, i) for i in stats.index]
        st.dataframe(
            stats.style.format("{:.2f}").background_gradient(cmap="Blues", subset=["mean"]),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────
# TAB 2 — Ranking Top 100 RBD por prueba
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🏆 Ranking Top 100 Establecimientos (RBD) por Promedio de Prueba")
    st.caption("Se excluyen puntajes igual a 0 antes de calcular el promedio por establecimiento. "
               "El promedio refleja solo a quienes efectivamente rindieron cada prueba.")

    all_cols = [cols_present[l] for l in selected_labels]

    # ── Controles ────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        prueba_rank = st.selectbox(
            "Prueba para ordenar el ranking",
            options=selected_labels,
            index=0
        )
    with c2:
        min_n = st.number_input(
            "Mínimo de estudiantes por RBD (filtra establecimientos pequeños)",
            min_value=1, value=10, step=5
        )
    with c3:
        highlight_rbd = st.checkbox(f"Resaltar RBD {int(rbd_input)}", value=True)

    prueba_col = cols_present[prueba_rank]

    # ── Calcular promedio por RBD (ceros → NaN antes de agrupar) ──
    df_no_zeros = df_2026[["RBD"] + all_cols].copy()
    df_no_zeros[all_cols] = df_no_zeros[all_cols].apply(lambda x: x.where(x != 0))

    # Promedio y conteo por RBD
    agg_mean = df_no_zeros.groupby("RBD")[all_cols].mean()
    agg_count = df_no_zeros.groupby("RBD")[all_cols].count()

    # Filtrar por mínimo de estudiantes en la prueba de ordenamiento
    valid_rbd = agg_count[prueba_col][agg_count[prueba_col] >= min_n].index
    agg_mean = agg_mean.loc[valid_rbd]
    agg_count = agg_count.loc[valid_rbd]

    # Renombrar columnas
    agg_mean.columns = [LABEL_MAP.get(c, c) for c in agg_mean.columns]
    agg_count.columns = [f"n_{LABEL_MAP.get(c, c)}" for c in agg_count.columns]

    friendly_prueba = LABEL_MAP.get(prueba_col, prueba_col)
    friendly_selected = [LABEL_MAP.get(c, c) for c in all_cols]

    df_ranking = pd.concat([agg_mean, agg_count], axis=1)
    df_ranking = (
        df_ranking
        .sort_values(friendly_prueba, ascending=False)
        .head(100)
        .reset_index()
    )
    df_ranking.index += 1
    df_ranking.index.name = "Posición"

    # Posición del RBD analizado
    rbd_pos = df_ranking[df_ranking["RBD"] == rbd_input].index.tolist()

    with c3:
        if rbd_pos:
            st.metric("Posición de tu RBD", f"#{rbd_pos[0]}")
        else:
            st.caption(f"RBD {int(rbd_input)} no aparece en el top 100\n(o no cumple el mínimo de n)")

    # ── Gráfico de barras horizontales ───────────────────────
    top_plot = min(30, len(df_ranking))
    df_plot = df_ranking.head(top_plot).copy()

    palette_bars = sns.color_palette(palette_option, top_plot)
    bar_colors = []
    for idx, row in df_plot.iterrows():
        if highlight_rbd and row["RBD"] == rbd_input:
            bar_colors.append("crimson")
        else:
            bar_colors.append(palette_bars[idx - 1])

    fig2, ax2 = plt.subplots(figsize=(10, max(6, top_plot * 0.35)))
    bars = ax2.barh(
        [f"RBD {int(r)}" for r in df_plot["RBD"]],
        df_plot[friendly_prueba],
        color=bar_colors
    )
    ax2.invert_yaxis()
    ax2.set_xlabel(f"Promedio — {friendly_prueba}")
    ax2.set_title(f"Top {top_plot} RBD — {friendly_prueba} (n ≥ {int(min_n)} por RBD)", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    # Etiquetas de valor
    for bar in bars:
        ax2.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}", va="center", fontsize=8
        )

    if highlight_rbd and rbd_input in df_plot["RBD"].values:
        ax2.text(
            0.99, 0.01, f"▶ RBD {int(rbd_input)} en rojo",
            transform=ax2.transAxes, ha="right", va="bottom",
            color="crimson", fontsize=9
        )

    plt.tight_layout()
    st.pyplot(fig2)

    # ── Tabla completa ───────────────────────────────────────
    st.subheader(f"Tabla — Top 100 RBD ordenados por {friendly_prueba}")

    score_cols = friendly_selected
    n_cols = [f"n_{p}" for p in friendly_selected]
    display_order = ["RBD"] + score_cols + n_cols
    display_order = [c for c in display_order if c in df_ranking.columns]

    def highlight_row(row):
        return ["background-color: #ffe0e0" if row["RBD"] == rbd_input else "" for _ in row]

    styled = (
        df_ranking[display_order]
        .style
        .format({c: "{:.1f}" for c in score_cols})
        .format({c: "{:.0f}" for c in n_cols})
        .background_gradient(cmap="YlGn", subset=[friendly_prueba])
    )
    if highlight_rbd:
        styled = styled.apply(highlight_row, axis=1)

    st.dataframe(styled, use_container_width=True, height=500)

    csv_out = df_ranking[display_order].to_csv(index=True).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar ranking como CSV",
        data=csv_out,
        file_name=f"ranking_top100_{prueba_rank.replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────────────────────
# TAB 3 — Datos Crudos
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"Datos crudos — RBD {int(rbd_input)}")
    if df_rbd.empty:
        st.warning("No hay datos para el RBD seleccionado.")
    else:
        st.caption(f"{len(df_rbd)} registros totales.")
        search = st.text_input("🔎 Buscar en cualquier columna")
        df_show = df_rbd.reset_index(drop=True)
        if search:
            mask = df_show.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            df_show = df_show[mask]
            st.caption(f"{len(df_show)} resultados para «{search}»")
        st.dataframe(df_show, use_container_width=True, height=500)
        csv_raw = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar datos filtrados como CSV",
            data=csv_raw,
            file_name=f"datos_rbd_{int(rbd_input)}.csv",
            mime="text/csv"
        )
