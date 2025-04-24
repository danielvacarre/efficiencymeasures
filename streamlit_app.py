import streamlit as st
from pandas import DataFrame

from models.efficiency.svf_eff import SVFEff
from views.data_config   import configure_dataset
from models.efficiency.dea import DEA
from models.efficiency.fdh import FDH
from views.svf_config    import configure_svf

st.set_page_config(layout="wide")
st.title("Estimación de Fronteras")
st.header("Carga datos, simula SVF y calcula eficiencias")

# 1) Subida y preparación de datos
nd = configure_dataset()
if nd is None or nd[0] is None:
    st.stop()

df, inputs, outputs = nd

# Sidebar: selección de métodos clásicos
st.sidebar.subheader("1. Métodos clásicos")
CLASSICAL_METHODS = [
    "DEA_RI", "DEA_RO", "DEA_DDF", "DEA_WA", "DEA_RUI", "DEA_RUO", "DEA_ERG",
    "FDH_RI", "FDH_RO", "FDH_DDF", "FDH_WA", "FDH_RUI", "FDH_RUO", "FDH_ERG"
]
selected_classical = st.sidebar.multiselect(
    "Elige métodos DEA/FDH",
    CLASSICAL_METHODS,
    default=["DEA_RI"]
)

# Sidebar: simulación SVF
st.sidebar.subheader("2. Simulación SVF")
svf = configure_svf(df, inputs, outputs)

# Sidebar: métodos SVF (solo si hay modelo entrenado)
SVF_METHODS = ["SVF_RI", "SVF_RI-", "SVF_RI+",
               "SVF_RO", "SVF_RO-", "SVF_RO+",
               "SVF_DDF", "SVF_DDF-", "SVF_DDF+",
               "SVF_WA", "SVF_WA-", "SVF_WA+",
               "SVF_RUI", "SVF_RUI-", "SVF_RUI+",
               "SVF_RUO", "SVF_RUO-", "SVF_RUO+",
               "SVF_ERG", "SVF_ERG-", "SVF_ERG+",
               "SVF_COST", "SVF_COST-", "SVF_COST+",]

selected_svf = []
if svf is not None:
    st.sidebar.subheader("3. Métodos SVF")
    selected_svf = st.sidebar.multiselect(
        "Elige métodos SVF",
        SVF_METHODS
    )

# Botón unificado para calcular todas las eficiencias
def calculate_all():
    df_res: DataFrame = df[inputs + outputs].copy()

    # Métodos clásicos
    for m in selected_classical:
        tipo, op = m.split("_", 1)
        key = op.lower()
        ModelClass = DEA if tipo == "DEA" else FDH
        model = ModelClass(inputs, outputs, df, methods=[key])
        func = getattr(model, f"calculate_{key}")
        print(f"Calculando {key} con {tipo}")
        df_res[m] = func()
        print(df_res)

    # Métodos SVF
    if svf is not None:
        df_est = svf.grid.data_grid
        weights = svf.solution.w
        for m in selected_svf:
            print(m)
            _, op = m.split("_", 1)
            # Detectar signo
            if op.endswith('+'):
                eps_val = svf.eps
                method_key = op[:-1].lower()
            elif op.endswith('-'):
                eps_val = -svf.eps
                method_key = op[:-1].lower()
            else:
                eps_val = 0.0
                method_key = op.lower()

            eff_solver = SVFEff(
                inputs=inputs,
                outputs=outputs,
                data=df,
                df_estimation=df_est,
                weights=weights,
                eps=svf.eps
            )
            func = getattr(eff_solver, f"calculate_{method_key}")
            # Pasamos eps_val sólo si es distinto de 0
            if eps_val != 0.0:
                df_res[m] = func(eps_val)
            else:
                df_res[m] = func()
    return df_res

if st.sidebar.button("▶ Calcular todas las eficiencias"):
    df_all = calculate_all()
    st.success("Cálculo completado")
    st.dataframe(df_all, use_container_width=True)