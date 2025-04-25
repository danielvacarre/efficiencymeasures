import streamlit as st
from pandas import DataFrame

from views.data_config    import configure_dataset
from views.model_config   import configure_svf
from views.results_config import calculate_all

st.set_page_config(layout="wide")
st.title("Estimación de Fronteras")
st.header("Carga datos, simula SVF y calcula eficiencias")

# 1) Subida y preparación de datos
nd = configure_dataset()
if nd is None or nd[0] is None:
    st.stop()

df, inputs, outputs, prices, weights = nd

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

if svf is not None:
    st.subheader("Visualizaciones y exportación SVF")

    if st.button("Mostrar frontera"):
        fig = svf.plot_frontier(num_points=200, show_data=True)
        st.plotly_chart(fig, use_container_width=True)
    # 3.3. Ver la solución (pesos)
    if st.button("Visualizar soluciones SVF"):
        # Asumo que svf.solution.w es array de pesos
        w = svf.solution.w
        xi = svf.solution.xi
        df_sol = DataFrame()

        if len(outputs) == 1:
            df_sol["w"] = w
            df_sol["xi"] = xi
        else:
            for j, out in enumerate(svf.outputs):
                df_sol[f"w_{out}"] = w[:, j]
                df_sol[f"xi_{out}"] = xi[:, j]
        st.dataframe(df_sol, use_container_width=True)


# Sidebar: métodos SVF (solo si hay modelo entrenado)
SVF_METHODS = ["SVF_RI", "SVF_RI-", "SVF_RI+",
               "SVF_RO", "SVF_RO-", "SVF_RO+",
               "SVF_DDF", "SVF_DDF-", "SVF_DDF+",
               "SVF_WA", "SVF_WA-", "SVF_WA+",
               "SVF_RUI", "SVF_RUI-", "SVF_RUI+",
               "SVF_RUO", "SVF_RUO-", "SVF_RUO+",
               "SVF_ERG", "SVF_ERG-", "SVF_ERG+",
               "CSVF_RI", "CSVF_RI-", "CSVF_RI+",
               "CSVF_RO", "CSVF_RO-", "CSVF_RO+",
               "CSVF_DDF", "CSVF_DDF-", "CSVF_DDF+",
               "CSVF_WA", "CSVF_WA-", "CSVF_WA+",
               "CSVF_RUI", "CSVF_RUI-", "CSVF_RUI+",
               "CSVF_RUO", "CSVF_RUO-", "CSVF_RUO+",
               "CSVF_ERG", "CSVF_ERG-", "CSVF_ERG+",
               "CSVF_COST", "CSVF_COST-", "CSVF_COST+",
               "CSVF_PROFIT", "CSVF_PROFIT-", "CSVF_PROFIT+",
               ]

selected_svf = []
if svf is not None:
    st.sidebar.subheader("3. Métodos SVF")
    selected_svf = st.sidebar.multiselect(
        "Elige métodos SVF",
        SVF_METHODS
    )


def calculate_asignative_profit(df, inputs, outputs, prices, weights, df_all):
    df_res: DataFrame = df[inputs + outputs + prices + weights].copy()

    pesos = df[weights]
    wx =  df[weights].mul(pesos.values).sum(axis=1)

    df_res["wx"] = wx
    df_res["CE"] = df_all["CSVF_COST"] /wx
    df_res["CE-"] = df_all["CSVF_COST-"] /wx
    df_res["CE+"] = df_all["CSVF_COST+"] /wx

    df_res["AE"] = df_res["CE"] / df_all["CSVF_RI"]
    df_res["AE-"] = df_res["CE-"] / df_all["CSVF_RI-"]
    df_res["AE+"] = df_res["CE+"] / df_all["CSVF_RI+"]

    prices = df[prices]
    py = df[outputs].mul(prices.values).sum(axis=1)

    df_res["py-wx"] = py - wx
    df_res["py+wx"] = py + wx

    df_res["PI"] = (df_all["CSVF_PROFIT"] - df_res["py-wx"]) / df_res["py+wx"]
    df_res["PI-"] = (df_all["CSVF_PROFIT-"] - df_res["py-wx"]) / df_res["py+wx"]
    df_res["PI+"] = (df_all["CSVF_PROFIT+"] - df_res["py-wx"]) / df_res["py+wx"]

    df_res["AI"] = df_res["PI"] / df_all["CSVF_DDF"]
    df_res["AI-"] = df_res["PI-"] / df_all["CSVF_DDF-"]
    df_res["AI+"] = df_res["PI+"] / df_all["CSVF_DDF+"]

    return df_res

if st.sidebar.button("▶ Calcular todas las eficiencias"):
    df_eff = calculate_all(df,inputs,outputs, selected_classical, selected_svf, svf, prices, weights)
    st.success("Cálculo completado")
    st.dataframe(df_eff, use_container_width=True)

    st.write("Eficiencia AE y AI")
    df_ae_ai = calculate_asignative_profit(df,inputs, outputs, prices, weights, df_eff)
    st.success("Cálculo completado")
    st.dataframe(df_ae_ai, use_container_width=True)