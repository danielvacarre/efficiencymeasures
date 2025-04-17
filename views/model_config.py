from streamlit import sidebar, multiselect

from models.dea import DEA
from models.fdh import FDH


def configure_model():

    # --- Model selection ---
    ALL_METHODS = [
        "DEA_RI", "DEA_RO", "DEA_DDF", "DEA_WA", "DEA_RUI", "DEA_RUO", "DEA_ERG",
        "FDH_RI", "FDH_RO", "FDH_DDF", "FDH_WA", "FDH_RUI", "FDH_RUO", "FDH_ERG"
    ]
    selected = multiselect(
        "Selecciona los métodos para calcular eficiencia",
        ALL_METHODS,
        default=["DEA_RI"]
    )
    return selected