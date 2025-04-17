# streamlit_app.py
import streamlit as st
from views.data_config import configure_dataset
from views.model_config import configure_model
from views.results_config import show_results

st.title("Estimación de Fronteras")
st.header("Carga datos, selecciona variables y ajusta tu modelo")

df, X_vars, Y_vars = configure_dataset()
if df is not None:
     model = configure_model(df, X_vars, Y_vars, key_suffix="modelo1")
     if model:
         with st.spinner("Entrenando modelo..."):
            show_results(model)