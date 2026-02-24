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

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Análisis de Puntajes — Admisión 2026")
st.markdown("---")

# --- Columnas disponibles ---
COLS_MAP = {
    "Comprensión Lectora": "CLEC_REG_ACTUAL",
    "Matemática 1":        "MATE1_REG_ACTUAL",
    "Matemática 2":        "MATE2_REG_ACTUAL",
    "Historia y Cs. Soc.": "HCSOC_REG_ACTUAL",
    "Ciencias":            "CIEN_REG_ACTUAL",
}
LABEL_MAP = {v: k for k, v in COLS_MAP.items()}

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuración")

    uploaded_file = st.file_uploader(
        "Cargar archivo CSV",
        type=["csv"],
        help="Ejemplo: ArchivoC_Adm2026REG.csv"
    )
    sep = st.selectbox("Separador del CSV", [";", ",", "|"], index=0)
    rbd_input = st.number_input("Filtrar por RBD", min_value=0, value=8998, step=1)
    palette_option = st.selectbox(
        "Paleta de colores",
        ["viridis", "magma", "plasma", "coolwarm", "Set2", "tab10"],
    )
    show_points = st.checkbox("Mostrar puntos individuales (stripplot)", value=False)
    st.markdown("---")
    st.caption("📌 Análisis de admisión universitaria Chile")

# --- Sin archivo ---
if uploaded_file is None:
    st.info("👈 Carga un archivo CSV desde el panel lateral para comenzar.")
    st.markdown("""
    ### ¿Cómo usar esta app?
    1. Sube el archivo `ArchivoC_Adm2026REG.csv` desde el panel lateral
    2. Ingresa el **RBD** del establecimiento a analizar
    3. Selecciona las pruebas y navega por las pestañas

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

# --- Cargar datos ---
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

df_rbd = df_2026[df_2026["RBD"] == rbd_input]
if df_rbd.empty:
    st.warning(f"⚠️ No se encontraron registros para RBD = {rbd_input}.")
    st.stop()

# Multiselect pruebas (depende de cols disponibles)
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

# Ceros → NaN (por columna, sin eliminar filas)
df_filtered = df_rbd[selected_cols].copy().apply(lambda x: x.where(x != 0))

# --- Métricas resumen ---
st.subheader(f"📋 Resumen — RBD {int(rbd_input)}")
st.caption(f"Total de estudiantes en el establecimiento: **{len(df_rbd)}**")
metric_cols = st.columns(len(selected_labels))
for i, (label, col) in enumerate(zip(selected_labels, selected_cols)):
    serie = df_filtered[col].dropna()
    metric_cols[i].metric(
        label=label,
        value=f"{serie.mean():.1f}" if len(serie) > 0 else "—",
        delta=f"n = {len(serie)}"
    )
st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📦 Distribución de Puntajes",
    "🏆 Ranking Top 100",
    "🔍 Datos Crudos",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — Boxplot
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Distribución de Puntajes por Prueba (ceros excluidos)")

    melted = (
        df_filtered
        .melt(var_name="Test", value_name="Puntaje")
        .dropna()
    )
    melted["Prueba"] = melted["Test"].map(LABEL_MAP)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=melted, x="Prueba", y="Puntaje",
        hue="Prueba", palette=palette_option,
        legend=False, ax=ax
    )
    if show_points:
        sns.stripplot(
            data=melted, x="Prueba", y="Puntaje",
            color="black", alpha=0.3, size=3, jitter=True, ax=ax
        )

    test_counts = df_filtered.count().to_dict()
    new_labels = [
        f"{LABEL_MAP.get(col, col)}\n(n={test_counts[col]})"
        for col in selected_cols
    ]
    ax.set_xticks(range(len(selected_cols)))
    ax.set_xticklabels(new_labels, rotation=30, ha="right")
    ax.set_title(
        f"Distribución de Puntajes — RBD {int(rbd_input)} (ceros excluidos)",
        fontsize=13, fontweight="bold"
    )
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
# TAB 2 — Ranking Top 100
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🏆 Ranking Top 100 — Promedio de Pruebas Rendidas (ceros excluidos)")

    all_cols = [cols_present[l] for l in selected_labels]

    # Identificadores opcionales
    id_cols = [c for c in ["MRUN", "RUN", "NOMBRE", "NOM_ALUMNO"] if c in df_rbd.columns]

    df_rank = df_rbd[id_cols + all_cols].copy().reset_index(drop=True)
    df_rank[all_cols] = df_rank[all_cols].apply(lambda x: x.where(x != 0))

    df_rank["Promedio"] = df_rank[all_cols].mean(axis=1, skipna=True)
    df_rank["Pruebas rendidas"] = df_rank[all_cols].notna().sum(axis=1)

    # Solo quienes rindieron al menos una
    df_rank = df_rank[df_rank["Pruebas rendidas"] > 0]

    # Filtro mínimo de pruebas
    col_slider, col_info = st.columns([2, 1])
    with col_slider:
        min_tests = st.slider(
            "Mínimo de pruebas rendidas para aparecer en el ranking",
            min_value=1, max_value=len(all_cols), value=1
        )

    df_rank_filtered = (
        df_rank[df_rank["Pruebas rendidas"] >= min_tests]
        .sort_values("Promedio", ascending=False)
        .head(100)
        .reset_index(drop=True)
    )
    df_rank_filtered.index += 1
    df_rank_filtered.index.name = "Ranking"

    with col_info:
        st.metric("Estudiantes en ranking", len(df_rank_filtered))

    st.caption(
        f"Mostrando hasta 100 estudiantes con al menos **{min_tests}** "
        f"prueba(s) rendida(s), ordenados por promedio descendente."
    )

    # Gráfico de barras top 50
    top_n = min(50, len(df_rank_filtered))
    fig2, ax2 = plt.subplots(figsize=(13, 4))
    colors = sns.color_palette(palette_option, top_n)
    ax2.bar(range(1, top_n + 1), df_rank_filtered["Promedio"].head(top_n), color=colors)
    ax2.set_xlabel("Posición en el ranking")
    ax2.set_ylabel("Promedio")
    ax2.set_title(
        f"Top {top_n} estudiantes por promedio — RBD {int(rbd_input)}",
        fontweight="bold"
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig2)

    # Tabla
    df_rank_filtered.rename(columns=LABEL_MAP, inplace=True)
    friendly_selected = [LABEL_MAP.get(c, c) for c in all_cols]
    display_cols = id_cols + friendly_selected + ["Promedio", "Pruebas rendidas"]
    display_cols = [c for c in display_cols if c in df_rank_filtered.columns]

    fmt = {c: "{:.1f}" for c in friendly_selected + ["Promedio"]}

    st.dataframe(
        df_rank_filtered[display_cols]
        .style
        .format(fmt)
        .background_gradient(cmap="YlGn", subset=["Promedio"])
        .bar(subset=["Promedio"], color="#90EE90"),
        use_container_width=True,
        height=460
    )

    csv_export = df_rank_filtered[display_cols].to_csv(index=True).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar Top 100 como CSV",
        data=csv_export,
        file_name=f"top100_rbd_{int(rbd_input)}.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────────────────────
# TAB 3 — Datos Crudos
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"Datos crudos — RBD {int(rbd_input)}")
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
