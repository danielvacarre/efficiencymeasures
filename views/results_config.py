from streamlit import button, success, dataframe
from pandas import DataFrame
from models.efficiency.dea import DEA
from models.efficiency.fdh import FDH

def show_results(
    selected_methods: list[str],
    df: DataFrame,
    inputs: list[str],
    outputs: list[str]
) -> None:
    if button("Calcular eficiencias DEA/FDH"):
        df_results = df[inputs + outputs].copy()
        for m in selected_methods:
            tipo, op = m.split("_", 1)
            method_key = op.lower()
            ModelClass = DEA if tipo == "DEA" else FDH
            model = ModelClass(inputs, outputs, df, methods=[method_key])
            func = getattr(model, f"calculate_{method_key}")
            df_results[m] = func()
        success("¡Listo!")
        dataframe(df_results)