from models.svf.svf_methods.ssvf import SSVF

def create_svf_method(method, inputs, outputs, df, C, eps, d, parallel):
    """
    Create an instance of the SVF method based on the provided method name.
    """
    if method == "SSVF":
        return SSVF(inputs, outputs, df, C, eps, d, parallel)
    else:
        raise ValueError(f"Unknown SVF method: {method}")