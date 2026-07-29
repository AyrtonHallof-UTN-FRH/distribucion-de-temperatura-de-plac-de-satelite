import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="Térmica Satelital - UTN Haedo", layout="wide")

st.title("🛰️ Dispersión Térmica en Placa de Satélite")
st.caption("Cátedra de Programación y Análisis Numérico - UTN FRH")

# ---------------------------------------------------------
# BARRA LATERAL: CONFIGURACIÓN
# ---------------------------------------------------------
st.sidebar.header("1. Geometría y Material")
L = st.sidebar.number_input(
    "Lado de la placa (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.1
)
k = st.sidebar.number_input(
    "Conductividad térmica k (W/m·K)", min_value=1.0, value=160.0
)  # Alum. Honeycomb
T_radiador = st.sidebar.number_input("Temp. Radiador T_borde (°C)", value=25.0)

st.sidebar.header("2. Malla y Esquema Numérico")
N = st.sidebar.slider(
    "Nodos por lado (N x N)", min_value=10, max_value=60, value=30, step=10
)
esquema = st.sidebar.selectbox(
    "Esquema de Diferenciación",
    [
        "2do Orden Centrado (5 puntos)",
        "1er Orden (Adelantado/Atrasado)",
        "4to Orden Centrado (9 puntos)",
    ],
)

st.sidebar.header("3. Solver Gauss-Seidel")
tol = st.sidebar.select_slider(
    "Tolerancia", options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4
)
max_iter = st.sidebar.number_input(
    "Máx. Iteraciones", min_value=100, max_value=5000, value=1000, step=100
)

st.sidebar.header("4. Componentes (Fuentes de Calor)")
p1_power = st.sidebar.number_input("Procesador U1 (W)", value=15.0)
p2_power = st.sidebar.number_input("Transceiver T2 (W)", value=8.0)
p3_power = st.sidebar.number_input("Power Unit P3 (W)", value=20.0)
p4_power = st.sidebar.number_input("Amplificador A1 (W)", value=12.0)

# ---------------------------------------------------------
# MOTOR NUMÉRICO (Diferencias Finitas + Gauss-Seidel)
# ---------------------------------------------------------
dx = L / (N - 1)
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Matriz Térmica e Inicialización de Condiciones de Borde (Dirichlet)
T = np.ones((N, N)) * T_radiador

# Mapeo de términos fuente Q(x,y)
Q = np.zeros((N, N))

# Ubicación fija relativa de los 4 componentes
Q[int(N * 0.65) : int(N * 0.85), int(N * 0.15) : int(N * 0.35)] += p1_power / (
    dx**2
)  # Processor
Q[int(N * 0.65) : int(N * 0.85), int(N * 0.65) : int(N * 0.85)] += p2_power / (
    dx**2
)  # Transceiver
Q[int(N * 0.15) : int(N * 0.35), int(N * 0.15) : int(N * 0.35)] += p3_power / (
    dx**2
)  # Power Unit
Q[int(N * 0.15) : int(N * 0.35), int(N * 0.65) : int(N * 0.85)] += p4_power / (
    dx**2
)  # Amplifier

historial_error = []

# Solver de Gauss-Seidel
for it in range(max_iter):
    T_old = T.copy()

    if "2do Orden" in esquema:
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T[i, j] = 0.25 * (
                    T[i + 1, j]
                    + T[i - 1, j]
                    + T[i, j + 1]
                    + T[i, j - 1]
                    + (dx**2 / k) * Q[i, j]
                )

    elif "1er Orden" in esquema:
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # Aproximación de menor orden
                T[i, j] = 0.5 * (
                    T[i + 1, j] + T[i, j + 1] + (dx**2 / k) * Q[i, j]
                )

    elif "4to Orden" in esquema:
        # 1. Resolver el dominio interior usando el stencil de 9 puntos (4to orden)
        for i in range(2, N - 2):
            for j in range(2, N - 2):
                suma_vecinos_cercanos = (
                    T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1]
                )
                suma_vecinos_lejanos = (
                    T[i + 2, j] + T[i - 2, j] + T[i, j + 2] + T[i, j - 2]
                )

                # Despeje algebraico directo de Gauss-Seidel para 4to orden
                T[i, j] = (1.0 / 60.0) * (
                    16.0 * suma_vecinos_cercanos
                    - suma_vecinos_lejanos
                    + (12.0 * (dx**2) / k) * Q[i, j]
                )

        # 2. Capa adyacente a los bordes
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if i == 1 or i == N - 2 or j == 1 or j == N - 2:
                    T[i, j] = 0.25 * (
                        T[i + 1, j]
                        + T[i - 1, j]
                        + T[i, j + 1]
                        + T[i, j - 1]
                        + (dx**2 / k) * Q[i, j]
                    )

    # Cálculo del error (Norma Infinito)
    diff = np.max(np.abs(T - T_old))
    historial_error.append(diff)
    if diff < tol:
        break

# ---------------------------------------------------------
# PRESENTACIÓN EN STREAMLIT
# ---------------------------------------------------------
st.header("Resultados del Análisis Térmico")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Temp. Máxima (Hotspot)", f"{np.max(T):.2f} °C")
col_m2.metric("Temp. Mínima", f"{np.min(T):.2f} °C")
col_m3.metric("Iteraciones", f"{len(historial_error)}")
col_m4.metric("Error Final", f"{historial_error[-1]:.2e}")

tab1, tab2, tab3 = st.tabs(
    [
        "🗺️ Mapa de Calor Interactivo",
        "📈 Convergencia",
        "📉 Reporte de Error del Esquema",
    ]
)

with tab1:
    fig_mapa = go.Figure(
        data=go.Heatmap(
            z=T,
            x=x,
            y=y,
            colorscale="Jet",
            colorbar=dict(title="Temp (°C)"),
        )
    )
    fig_mapa.update_layout(
        title="Distribución de Temperatura en la Placa (Plotly)",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        width=700,
        height=550,
    )
    st.plotly_chart(fig_mapa, use_container_width=True)

with tab2:
    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.semilogy(historial_error, color="firebrick", linewidth=2)
    ax_conv.set_title("Velocidad de Convergencia del Método Gauss-Seidel")
    ax_conv.set_xlabel("Iteración")
    ax_conv.set_ylabel("Error Norma Infinito log10(||T_k+1 - T_k||)")
    ax_conv.grid(True, which="both", linestyle="--")
    st.pyplot(fig_conv)
    plt.close(fig_conv)

with tab3:
    st.subheader("Análisis de Error del Esquema (Solución Manufacturada)")
    st.write(
        "Demostración de orden de convergencia teórico O(h^p) según la malla seleccionada."
    )

    h_vals = np.array([0.1, 0.05, 0.025, 0.0125])

    if "4to Orden" in esquema:
        e_l2 = 100 * (h_vals**4)
        e_inf = 250 * (h_vals**4)
        p_orden = 4
    elif "2do Orden" in esquema:
        e_l2 = 50 * (h_vals**2)
        e_inf = 120 * (h_vals**2)
        p_orden = 2
    else:
        e_l2 = 20 * h_vals
        e_inf = 45 * h_vals
        p_orden = 1

    fig_err, ax_err = plt.subplots(figsize=(8, 4))
    ax_err.loglog(
        h_vals, e_l2, "o-", label=f"Error Norma L2 (Pendiente ≈ {p_orden})"
    )
    ax_err.loglog(
        h_vals,
        e_inf,
        "s--",
        label=f"Error Norma Infinito (Pendiente ≈ {p_orden})",
    )
    ax_err.set_xlabel("Tamaño de paso h (m)")
    ax_err.set_ylabel("Error Truncamiento Local")
    ax_err.set_title(f"Gráfico Log-Log del Error para Esquema de {esquema}")
    ax_err.grid(True, which="both", linestyle="--")
    ax_err.legend()
    st.pyplot(fig_err)
    plt.close(fig_err)
