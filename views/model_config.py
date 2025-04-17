from streamlit import sidebar

from models.fdh import FDH


def configure_model(df, X_vars, Y_vars, key_suffix=""):
    # --- Model selection ---
    alg = sidebar.selectbox(
        "Select algorithm:",["FDH", "DEA"],
    key = f"algorithm_selectbox_{key_suffix}"
    )

    if alg == "FDH":
    # Available FDH methods
        available_methods = [
            "ri",  # Input-oriented
            "ro",  # Output-oriented
            "ddf",  # Directional Distance
            "wa", "rui", "ruo", "erg"  # Additive & Russell
        ]

    methods = sidebar.multiselect(
        "FDH methods:",
        available_methods,
        default=[available_methods[0]],
        key=f"methods_multiselect_{key_suffix}"
    )

    # Initialize FDH model with chosen methods
    model = FDH(
        inputs=X_vars,
        outputs=Y_vars,
        data=df,
        methods=methods
    )
    return model
