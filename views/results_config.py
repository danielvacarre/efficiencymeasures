# views/results_view.py

import streamlit as st
from pandas import DataFrame
from typing import Optional

from models.efficiency_method import EfficiencyMethod


def show_results(
    model: EfficiencyMethod,
    eps: Optional[float] = 0.0) -> None:
    """
    Render a button to compute efficiencies and display them as a DataFrame.

    Args:
        model: An EfficiencyMethod (or subclass) instance, already configured
               with inputs, outputs and selected methods.
        eps:   Epsilon parameter for methods that require it (default 0.0).
    """
    if st.button("Calculate Efficiencies"):
        with st.spinner("Calculating efficiencies…"):
            # This will return the original inputs+outputs plus one column per method
            df_eff: DataFrame = model.get_efficiencies(eps)
        st.success("Done!")
        st.dataframe(df_eff)