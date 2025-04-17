from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize(df, method="standard"):
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    cols = df.columns
    df_scaled = scaler.fit_transform(df)
    return DataFrame(df_scaled, columns=cols)