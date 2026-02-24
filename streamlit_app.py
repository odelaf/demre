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
    2. Ingresa el **RBD** del establecimiento a analizar en la pestaña de distribución
    3. En **Ranking por Prueba** compara establecimientos en una prueba específica
    4. En **Ranking General** compara por el promedio de todos los promedios

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
df_filtered = (
    df_rbd[selected_cols].copy().apply(lambda x: x.where(x != 0))
    if not df_rbd.empty else pd.DataFrame(columns=selected_cols)
)

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

# ── Helper: construir tabla de promedios por RBD ─────────────
@st.cache_data
def build_rbd_averages(data_bytes, sep, all_cols_raw):
    import io
    df = pd.read_csv(io.BytesIO(data_bytes), sep=sep, low_memory=False)
    df_nz = df[["RBD"] + all_cols_raw].copy()
    df_nz[all_cols_raw] = df_nz[all_cols_raw].apply(lambda x: x.where(x != 0))
    agg_mean  = df_nz.groupby("RBD")[all_cols_raw].mean()
    agg_count = df_nz.groupby("RBD")[all_cols_raw].count()
    return agg_mean, agg_count

uploaded_file.seek(0)
raw_bytes = uploaded_file.read()
all_cols_raw = [cols_present[l] for l in selected_labels]
agg_mean, agg_count = build_rbd_averages(raw_bytes, sep, all_cols_raw)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Distribución de Puntajes",
    "🏆 Ranking por Prueba",
    "🥇 Ranking General",
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
            [f"{LABEL_MAP.get(c,c)}\n(n={counts[c]})" for c in selected_cols],
            rotation=30, ha="right"
        )
        ax.set_title(f"Distribución de Puntajes — RBD {int(rbd_input)}", fontsize=13, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("Puntaje")
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
# TAB 2 — Ranking Top 100 por prueba específica
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🏆 Ranking Top 100 Establecimientos por Prueba (ceros excluidos)")
    st.caption("Promedio calculado solo sobre alumnos que rindieron efectivamente cada prueba (puntaje > 0).")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        prueba_rank = st.selectbox("Prueba para ordenar el ranking", options=selected_labels, index=0)
    with c2:
        min_n_t2 = st.number_input("Mínimo de estudiantes por RBD", min_value=1, value=10, step=5, key="min_n_t2")
    with c3:
        highlight_t2 = st.checkbox(f"Resaltar RBD {int(rbd_input)}", value=True, key="hl_t2")

    prueba_col = cols_present[prueba_rank]
    friendly_prueba = LABEL_MAP[prueba_col]

    valid_rbd_t2 = agg_count[prueba_col][agg_count[prueba_col] >= min_n_t2].index
    am_t2 = agg_mean.loc[valid_rbd_t2].copy()
    ac_t2 = agg_count.loc[valid_rbd_t2].copy()
    am_t2.columns = [LABEL_MAP[c] for c in am_t2.columns]
    ac_t2.columns = [f"n_{LABEL_MAP[c]}" for c in ac_t2.columns]

    df_rank_t2 = pd.concat([am_t2, ac_t2], axis=1).sort_values(friendly_prueba, ascending=False).head(100).reset_index()
    df_rank_t2.index += 1
    df_rank_t2.index.name = "Posición"

    rbd_pos_t2 = df_rank_t2[df_rank_t2["RBD"] == rbd_input].index.tolist()
    with c3:
        if rbd_pos_t2:
            st.metric("Tu RBD", f"#{rbd_pos_t2[0]}")
        else:
            st.caption(f"RBD {int(rbd_input)} fuera del top 100")

    # Gráfico barras horizontales
    top_plot = min(30, len(df_rank_t2))
    df_plot = df_rank_t2.head(top_plot)
    bar_colors = [
        "crimson" if (highlight_t2 and r == rbd_input) else sns.color_palette(palette_option, top_plot)[i]
        for i, r in enumerate(df_plot["RBD"])
    ]
    fig2, ax2 = plt.subplots(figsize=(10, max(6, top_plot * 0.35)))
    bars = ax2.barh([f"RBD {int(r)}" for r in df_plot["RBD"]], df_plot[friendly_prueba], color=bar_colors)
    ax2.invert_yaxis()
    ax2.set_xlabel(f"Promedio — {friendly_prueba}")
    ax2.set_title(f"Top {top_plot} RBD — {friendly_prueba} (n ≥ {int(min_n_t2)})", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{bar.get_width():.1f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)

    friendly_sel = [LABEL_MAP[c] for c in all_cols_raw]
    n_cols_t2    = [f"n_{p}" for p in friendly_sel]
    disp_t2      = ["RBD"] + friendly_sel + n_cols_t2
    disp_t2      = [c for c in disp_t2 if c in df_rank_t2.columns]

    def hl_row_t2(row):
        return ["background-color:#ffe0e0" if row["RBD"] == rbd_input else "" for _ in row]

    styled_t2 = (
        df_rank_t2[disp_t2].style
        .format({c: "{:.1f}" for c in friendly_sel})
        .format({c: "{:.0f}" for c in n_cols_t2})
        .background_gradient(cmap="YlGn", subset=[friendly_prueba])
    )
    if highlight_t2:
        styled_t2 = styled_t2.apply(hl_row_t2, axis=1)

    st.dataframe(styled_t2, use_container_width=True, height=500)
    st.download_button(
        "⬇️ Descargar ranking como CSV",
        df_rank_t2[disp_t2].to_csv(index=True).encode("utf-8"),
        file_name=f"ranking_prueba_{prueba_rank.replace(' ','_')}.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────────────────────
# TAB 3 — Ranking General: promedio del promedio
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🥇 Ranking General — Top 100 por Promedio de Promedios (ceros excluidos)")
    st.caption(
        "Para cada establecimiento se calcula el promedio de cada prueba (excluyendo ceros), "
        "luego se promedian esos valores para obtener un **índice general**. "
        "Solo se incluyen pruebas donde el establecimiento tenga al menos el mínimo de alumnos exigido."
    )

    c1g, c2g, c3g = st.columns([2, 2, 1])
    with c1g:
        min_n_g = st.number_input("Mínimo de estudiantes por prueba por RBD", min_value=1, value=10, step=5, key="min_n_g")
    with c2g:
        min_pruebas = st.slider("Mínimo de pruebas válidas para incluir el RBD",
                                min_value=1, max_value=len(all_cols_raw), value=1)
    with c3g:
        highlight_g = st.checkbox(f"Resaltar RBD {int(rbd_input)}", value=True, key="hl_g")

    # Enmascarar promedios donde n < mínimo → NaN
    am_g = agg_mean.copy()
    for col in all_cols_raw:
        am_g.loc[agg_count[col] < min_n_g, col] = np.nan

    # Promedio del promedio (solo sobre pruebas válidas)
    am_g["Prom. General"] = am_g[all_cols_raw].mean(axis=1, skipna=True)
    am_g["Pruebas válidas"] = am_g[all_cols_raw].notna().sum(axis=1)

    # Filtrar por mínimo de pruebas válidas
    am_g = am_g[am_g["Pruebas válidas"] >= min_pruebas]

    # Renombrar cols para mostrar
    am_g_show = am_g.copy()
    am_g_show.columns = [LABEL_MAP.get(c, c) for c in am_g_show.columns]
    friendly_sel_g = [LABEL_MAP[c] for c in all_cols_raw]

    # Añadir n por prueba
    ac_g = agg_count.loc[am_g.index].copy()
    ac_g.columns = [f"n_{LABEL_MAP[c]}" for c in ac_g.columns]
    n_cols_g = list(ac_g.columns)

    df_general = (
        pd.concat([am_g_show, ac_g], axis=1)
        .sort_values("Prom. General", ascending=False)
        .head(100)
        .reset_index()
    )
    df_general.index += 1
    df_general.index.name = "Posición"

    rbd_pos_g = df_general[df_general["RBD"] == rbd_input].index.tolist()
    with c3g:
        if rbd_pos_g:
            st.metric("Tu RBD", f"#{rbd_pos_g[0]}")
            st.caption(f"Prom. General: {df_general.loc[rbd_pos_g[0], 'Prom. General']:.1f}")
        else:
            st.caption(f"RBD {int(rbd_input)} fuera del top 100")

    # ── Gráfico barras horizontales ───────────────────────────
    top_plot_g = min(30, len(df_general))
    df_plot_g  = df_general.head(top_plot_g)
    bar_colors_g = [
        "crimson" if (highlight_g and r == rbd_input) else sns.color_palette(palette_option, top_plot_g)[i]
        for i, r in enumerate(df_plot_g["RBD"])
    ]

    fig3, ax3 = plt.subplots(figsize=(10, max(6, top_plot_g * 0.35)))
    bars3 = ax3.barh(
        [f"RBD {int(r)}" for r in df_plot_g["RBD"]],
        df_plot_g["Prom. General"],
        color=bar_colors_g
    )
    ax3.invert_yaxis()
    ax3.set_xlabel("Promedio General")
    ax3.set_title(
        f"Top {top_plot_g} RBD — Promedio de Promedios (n ≥ {int(min_n_g)} por prueba, "
        f"≥ {min_pruebas} prueba(s) válida(s))",
        fontweight="bold", fontsize=11
    )
    ax3.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars3:
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{bar.get_width():.1f}", va="center", fontsize=8)
    if highlight_g and rbd_input in df_plot_g["RBD"].values:
        ax3.text(0.99, 0.01, f"▶ RBD {int(rbd_input)} en rojo",
                 transform=ax3.transAxes, ha="right", va="bottom",
                 color="crimson", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)

    # ── Tabla ─────────────────────────────────────────────────
    disp_g = ["RBD"] + friendly_sel_g + ["Prom. General", "Pruebas válidas"] + n_cols_g
    disp_g = [c for c in disp_g if c in df_general.columns]

    def hl_row_g(row):
        return ["background-color:#ffe0e0" if row["RBD"] == rbd_input else "" for _ in row]

    styled_g = (
        df_general[disp_g].style
        .format({c: "{:.1f}" for c in friendly_sel_g + ["Prom. General"]})
        .format({c: "{:.0f}" for c in n_cols_g + ["Pruebas válidas"]})
        .background_gradient(cmap="YlOrRd", subset=["Prom. General"])
    )
    if highlight_g:
        styled_g = styled_g.apply(hl_row_g, axis=1)

    st.dataframe(styled_g, use_container_width=True, height=500)
    st.download_button(
        "⬇️ Descargar ranking general como CSV",
        df_general[disp_g].to_csv(index=True).encode("utf-8"),
        file_name="ranking_general_top100.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────────────────────
# TAB 4 — Datos Crudos
# ─────────────────────────────────────────────────────────────
with tab4:
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
        st.download_button(
            "⬇️ Descargar datos filtrados como CSV",
            df_show.to_csv(index=False).encode("utf-8"),
            file_name=f"datos_rbd_{int(rbd_input)}.csv",
            mime="text/csv"
        )
