from matplotlib.pyplot import subplots
from matplotlib.pyplot import close
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="Térmica Satelital - UTN Haedo", layout="wide")

st.title("Dispersión Térmica en Placa de Satélite")
st.caption("Cátedra de Programación y Análisis Numérico - UTN FRH")

# ---------------------------------------------------------
# BARRA LATERAL: CONFIGURACIÓN
# ---------------------------------------------------------
st.sidebar.header("1. Geometría y Material")
L = st.sidebar.number_input(
    "Lado de la placa (m)", min_value=0.001, value=0.1
)

k = st.sidebar.number_input(
    "Conductividad térmica k (W/m·K)", min_value=0.01, value=130.0
)  #Aluminio 7075-T6

T_borde = st.sidebar.number_input("Temperatura de borde por radiador (°C)", value=25.0)

st.sidebar.header("2. Malla y Esquema Numérico")
N = st.sidebar.slider(
    "Nodos por lado (N x N)", min_value=10, max_value=100, value=50, step=10
)

esquema = st.sidebar.selectbox(
    "Esquema de Diferenciación",
    [
        "2do Orden Centrado (5 puntos)",
        "1er Orden Adelantado",
        "1er Orden Atrasado",
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

# ---------------------------------------------------------
# 4. COMPONENTES DINÁMICOS (FUENTES DE CALOR)
# ---------------------------------------------------------
st.sidebar.header("4. Componentes (Fuentes de Calor)")

# Permite agregar o quitar componentes cambiando la cantidad
cant_componentes = st.sidebar.number_input(
    "Cantidad de Componentes", min_value=1, value=4, step=1
)

componentes = []
# Posiciones por defecto predefinidas para no arrancar en cero
pos_defecto = [(75.0, 25.0), (75.0, 75.0), (25.0, 25.0), (25.0, 75.0)]

for idx in range(int(cant_componentes)):
    with st.sidebar.expander(f"Componente {idx + 1}"):
        p_w = st.number_input(
            f"Potencia (W) #{idx + 1}", value=15.0, key=f"p_{idx}"
        )
        if idx < len(pos_defecto):
          def_x, def_y = pos_defecto[idx] 
        else:
          def_x, def_y = (50.0, 50.0)

        px = st.number_input(
            f"Posición X (% de L) #{idx + 1}",
            value=def_x,
            min_value=0.0,
            max_value=100.0,
            key=f"px_{idx}"
        )
        py = st.number_input(
            f"Posición Y (% de L) #{idx + 1}",
            value=def_y,
            min_value=0.0,
            max_value=100.0,
            key=f"py_{idx}"
        )

        # Input para el tamaño del parche (porcentaje del lado L)
        tamano_pct = st.number_input(
            f"Tamaño (% de L) #{idx + 1}",
            value=15.0,
            min_value=1.0,
            max_value=100.0,
            key=f"tamano_{idx}"
        )

        # Agregamos 'tamano_pct' al diccionario del componente
        componentes.append({"potencia": p_w, "x_pct": px, "y_pct": py, "tamano_pct": tamano_pct})

# Botón de ejecución en la barra lateral
st.sidebar.markdown("---")
ejecutar_simulacion = st.sidebar.button("Ejecutar Simulación", type="primary", use_container_width=True)

# ---------------------------------------------------------
# MOTOR NUMÉRICO (Diferencias Finitas + Gauss-Seidel)
# ---------------------------------------------------------
if ejecutar_simulacion:
  dx = L / (N - 1)
  x = np.linspace(0, L, N)
  y = np.linspace(0, L, N)
  X, Y = np.meshgrid(x, y)

  # Matriz Térmica e Inicialización de Condiciones de Borde (Dirichlet)
  T = np.ones((N, N)) * T_borde

  # Mapeo de términos fuente Q(x,y)
  Q = np.zeros((N, N))

  # Mapeo dinámico de todas las fuentes definidas
  for comp in componentes:
          # NUEVO: Calculamos el ancho del parche específico para este componente
          ancho_parche = max(1, int(N * (comp["tamano_pct"] / 100.0)))

          # Convertir % de posición a índices de matriz
          idx_y = int((comp["x_pct"] / 100.0) * (N - 1))
          idx_x = int((comp["y_pct"] / 100.0) * (N - 1))

          # Delimitar rango de la celda evitando salir de los bordes
          i_min = max(1, idx_x - ancho_parche // 2)
          i_max = min(N - 1, idx_x + ancho_parche // 2 + 1)
          j_min = max(1, idx_y - ancho_parche // 2)
          j_max = min(N - 1, idx_y + ancho_parche // 2 + 1)

          # Calcular densidad volumétrica de calor Q [W/m^2]
          num_nodos = (i_max - i_min) * (j_max - j_min)
          if num_nodos > 0:
              Q[i_min:i_max, j_min:j_max] += (comp["potencia"] / num_nodos) / (dx**2)

  historial_error = []

  # Solver de Gauss-Seidel
  for it in range(int(max_iter)):
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

      elif "1er Orden Adelantado" in esquema:
          for i in range(1, N - 2):
              for j in range(1, N - 2):
                  T[i, j] = 0.5 * (
                      2.0 * T[i + 1, j] - T[i + 2, j]
                      + 2.0 * T[i, j + 1] - T[i, j + 2]
                      - (dx**2 / k) * Q[i, j]
                  )

          # 2. Capa adyacente al borde superior/derecho
          for i in range(1, N - 1):
              for j in range(1, N - 1):
                  if i == N - 2 or j == N - 2:
                      T[i, j] = 0.25 * (
                          T[i + 1, j]
                          + T[i - 1, j]
                          + T[i, j + 1]
                          + T[i, j - 1]
                          + (dx**2 / k) * Q[i, j]
                      )

      elif "1er Orden Atrasado" in esquema:
          for i in range(2, N - 1):
              for j in range(2, N - 1):
                  T[i, j] = 0.5 * (
                      2.0 * T[i - 1, j] - T[i - 2, j]
                      + 2.0 * T[i, j - 1] - T[i, j - 2]
                      + (dx**2 / k) * Q[i, j]
                  )

          # 2. Capa adyacente al borde (i=1 o j=1)
          for i in range(1, N - 1):
              for j in range(1, N - 1):
                  if i == 1 or j == 1:
                      T[i, j] = 0.25 * (
                          T[i + 1, j]
                          + T[i - 1, j]
                          + T[i, j + 1]
                          + T[i, j - 1]
                          + (dx**2 / k) * Q[i, j]
                      )

      elif "4to Orden" in esquema:
          for i in range(2, N - 2):
            for j in range(2, N - 2):
              suma_vecinos_cercanos = (
                      T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1]
                  )
              suma_vecinos_lejanos = (
                      T[i + 2, j] + T[i - 2, j] + T[i, j + 2] + T[i, j - 2]
                  )

              T[i, j] = (1.0 / 60.0) * (
                      16.0 * suma_vecinos_cercanos
                      - suma_vecinos_lejanos
                      + (12.0 * (dx**2) / k) * Q[i, j]
                  )

      # 2. CAPA DE PROTECCIÓN: Nodos i=1, j=1, i=N-2, j=N-2
      # Como no pueden mirar a i-2 o i+2, se resuelven con 2do orden centrado
          for i in range(1, N - 1):
            for j in range(1, N - 1):
              if i in (1, N - 2) or j in (1, N - 2):
                  T[i, j] = 0.25 * (
                      T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1]
                      + (dx**2 / k) * Q[i, j]
                  )

      # Cálculo del error
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

  tab1, tab2= st.tabs(
      [
          "Mapa de Calor Interactivo",
          "Convergencia",
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
          height=700,
      )
      st.plotly_chart(fig_mapa, use_container_width=True)

  with tab2:
      fig_conv, ax_conv = subplots(figsize=(8, 4))
      ax_conv.semilogy(historial_error, color="firebrick", linewidth=2)
      ax_conv.set_title("Velocidad de Convergencia del Método Gauss-Seidel")
      ax_conv.set_xlabel("Iteración")
      ax_conv.set_ylabel("Error")
      ax_conv.grid(True, which="both", linestyle="--")
      st.pyplot(fig_conv)
      close(fig_conv)

else:
    st.info("Ajustá la geometría, la malla y las fuentes de calor en la barra lateral. Luego presioná **Ejecutar Simulación** para ver los resultados.")
