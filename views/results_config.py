# views/results_view.py

import streamlit as st
from pandas import DataFrame
from typing import Optional

from models.dea import DEA
from models.efficiency_method import EfficiencyMethod
from models.fdh import FDH


def show_results(selected_methods: list[str], df: DataFrame, inputs: list[str], outputs: list[str]) -> None:

    if st.button("Calcular eficiencias"):
        df_results = df[inputs + outputs].copy()
        for m in selected_methods:
            tipo, op = m.split("_", 1)  # e.g. "DEA", "RI"
            method_key = op.lower()  # "ri", "ro", ...
            ModelClass = DEA if tipo == "DEA" else FDH
            model = ModelClass(inputs, outputs, df, methods=[method_key])
            func = getattr(model, f"calculate_{method_key}")
            eff_values = func()
            df_results[m] = eff_values
        st.success("¡Listo!")
        st.dataframe(df_results)