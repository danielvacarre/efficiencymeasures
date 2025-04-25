from streamlit import button, success, dataframe
from pandas import DataFrame

from models.efficiency.csvf_eff import CSVFEff
from models.efficiency.dea import DEA
from models.efficiency.fdh import FDH
from models.efficiency.svf_eff import SVFEff


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

# Botón unificado para calcular todas las eficiencias
def calculate_all(df,inputs,outputs, selected_classical, selected_svf, svf=None, prices=None, weights=None) -> DataFrame:
    df_res: DataFrame = df[inputs + outputs].copy()
    # Métodos clásicos
    for m in selected_classical:
        tipo, op = m.split("_", 1)
        key = op.lower()
        ModelClass = DEA if tipo == "DEA" else FDH
        model = ModelClass(inputs, outputs, df, methods=[key])
        func = getattr(model, f"calculate_{key}")
        df_res[m] = func()


    # Métodos SVF
    if svf is not None:
        svf.get_virtual_grid_estimation()
        for m in selected_svf:
            _, op = m.split("_", 1)
            if _ == "SVF":
                eff_solver = SVFEff(
                    inputs=inputs,
                    outputs=outputs,
                    data=df,
                    df_estimation=svf.grid.virtual_grid,
                    eps=svf.eps
                )
            else:
                eff_solver = CSVFEff(
                    inputs=inputs,
                    outputs=outputs,
                    data=df,
                    df_estimation=svf.grid.virtual_grid,
                    weights_cols=weights,
                    prices_cols=prices,
                    eps=svf.eps
                )
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


            func = getattr(eff_solver, f"calculate_{method_key}")
            # Pasamos eps_val sólo si es distinto de 0
            if eps_val != 0.0:
                df_res[m] = func(eps_val)
            else:
                df_res[m] = func()
    return df_res