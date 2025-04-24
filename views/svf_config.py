import streamlit as st
from numpy import trunc

from models.svf.svf_utils.inizialization import create_svf_method


def configure_svf(df, inputs, outputs):
    st.subheader("Configuración SVF")
    num_rows = len(df)
    method = st.selectbox("Método SVF", ["SSVF"])
    C      = st.number_input("Regularization C", value=1.0, format="%.3f")
    eps    = st.number_input("Epsilon",         value=0.0, format="%.3f", min_value=0.0)
    d      = st.slider("Grid partitions (d)", 1, int(num_rows), int(trunc(num_rows/2)))
    parallel = st.number_input("Workers", 1, 16, 4)

    if "svf_model" not in st.session_state:
        st.session_state.svf_model = None

    if st.button("Simular SVF"):
        with st.spinner("Entrenando SSVF…"):
            st.session_state.svf_model = create_svf_method(method, inputs, outputs, df, C, eps, d, parallel)
            st.session_state.svf_model.train()
            st.session_state.svf_model.solve()
        st.success(
            f"Entrenado en {st.session_state.svf_model.train_time.total_seconds():.2f} s"
        )

    return st.session_state.svf_model