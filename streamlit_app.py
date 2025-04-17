# streamlit_app.py
import streamlit as st

from views.data_config import configure_dataset
from views.model_config import configure_model
from views.results_config import show_results

st.title("Estimación de Fronteras")
st.header("Carga datos, selecciona variables y métodos")

# 1) Sube y prepara el dataset
df, inputs, outputs = configure_dataset()

if df is not None:
    # 2) Permite seleccionar uno o varios métodos
    selected_methods = configure_model()

    if selected_methods:
        # 3) Muestra botón y resultados
        show_results(selected_methods, df, inputs, outputs)
    else:
        st.info("Selecciona al menos un método para ver las eficiencias.")