from pandas import read_csv
from streamlit import sidebar, write
from utils.preprocessing import normalize

def configure_dataset():
    uploaded = sidebar.file_uploader("Sube tu CSV o TXT", type=["csv","txt"])
    has_header = sidebar.checkbox("¿Tiene encabezado?", True)
    delimiter  = sidebar.selectbox("Delimitador", [",",";","\t"," "], index=1)
    norm_method= sidebar.selectbox("Normalización", ["ninguna","standard","minmax"])
    if not uploaded:
        return None, None, None

    df = read_csv(uploaded, sep=delimiter, header=(0 if has_header else None))
    if not has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        write(f"Sin encabezado: {df.shape[1]} columnas detectadas")

    write("Vista previa:")
    write(df.head())
    cols = df.columns.tolist()
    inputs  = sidebar.multiselect("Variables de entrada (X)", cols)
    outputs = sidebar.multiselect("Variables de salida (Y)", [c for c in cols if c not in inputs])
    prices = sidebar.multiselect("Precios", [c for c in cols if c not in inputs and c not in outputs])
    weights = sidebar.multiselect("Pesos", [c for c in cols if c not in inputs and c not in outputs and c not in prices])


    if inputs and outputs:
        X = df[inputs]
        if norm_method != "ninguna":
            X = normalize(X, method=norm_method)
        df_pre = X.join(df[outputs])
        write("Datos listos para modelado")
        write(df_pre.head())

        df_pre = df_pre.join(df[prices])
        df_pre = df_pre.join(df[weights])
        return df_pre, inputs, outputs, prices, weights

    return None, None, None