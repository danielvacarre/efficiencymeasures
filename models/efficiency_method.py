from pandas import DataFrame
from typing import List, Optional


class EfficiencyMethod:
    """
    Class to calculate efficiencies using various frontier estimation methods.

    Parameters:
        inputs (List[str]): List of input variable column names.
        outputs (List[str]): List of output variable column names.
        data (pd.DataFrame): DataFrame containing the original data.
        methods (List[str]): Methods to compute efficiencies (e.g. 'ri', 'ddf', 'wa', etc.).
        df_estimation (Optional[pd.DataFrame]): Additional DataFrame for estimation (optional).
        weights (Optional[dict]): Custom weights for certain methods (optional).
        prices (Optional[object]): Prices for certain methods (optional).
    """
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        data: DataFrame,
        methods: List[str],
        df_estimation: Optional[DataFrame] = None,
        weights: Optional[dict] = None,
        prices: Optional[object] = None
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.data = data.copy()
        self.methods = methods
        self.df_estimation = df_estimation
        self.weights = weights
        self.prices = prices
        self.df_eff: Optional[DataFrame] = None

    def get_efficiencies(self, eps: float = 0.0) -> DataFrame:
        """
        Compute efficiencies for the specified methods and return a DataFrame.

        Args:
            eps (float): Epsilon value for methods that require it.

        Returns:
            DataFrame: DataFrame including input and output variables and efficiency columns.
        """
        # Copy only the relevant input and output columns
        df_eff = self.data[self.inputs + self.outputs].copy()

        # Iterate over each specified method and calculate its efficiency
        for method in self.methods:
            func = self._resolve_method_function(method)
            # Use eps parameter if required by the method
            if self._requires_eps(method):
                df_eff[method] = func(eps)
            else:
                df_eff[method] = func()

        self.df_eff = df_eff
        return df_eff

    def _resolve_method_function(self, method: str):
        """
        Map method name to the corresponding calculation function.
        """
        mapping = {
            "ri": self.calculate_ri,
            "ri-": self.calculate_ri,
            "ri+": self.calculate_ri,
            "ro": self.calculate_ro,
            "ddf": self.calculate_ddf,
            "ddf-": self.calculate_ddf,
            "ddf+": self.calculate_ddf,
            "wa": self.calculate_wa,
            "rui": self.calculate_rui,
            "ruo": self.calculate_ruo,
            "erg": self.calculate_erg,
            "c(y,w)": self.calculate_cost_model,
            "c(y,w)-": self.calculate_cost_model,
            "c(y,w)+": self.calculate_cost_model,
            "p(w,p)": self.calculate_profit_model,
            "p(w,p)-": self.calculate_profit_model,
            "p(w,p)+": self.calculate_profit_model,
        }
        if method not in mapping:
            raise ValueError(f"Unknown method: {method}")
        return mapping[method]

    def _requires_eps(self, method: str) -> bool:
        """
        Check if the specified method requires an epsilon parameter.
        """
        return method.endswith('-') or method.endswith('+')

    def calculate_ri(self, eps: float = 0.0) -> List[float]:
        """
        Compute input-oriented efficiency (RI) with optional epsilon shift.
        """
        raise NotImplementedError("calculate_ri must be implemented")

    def calculate_ro(self) -> List[float]:
        """
        Compute output-oriented efficiency (RO).
        """
        raise NotImplementedError("calculate_ro must be implemented")

    def calculate_ddf(self, eps: float = 0.0) -> List[float]:
        """
        Compute directional distance function efficiency (DDF) with optional epsilon.
        """
        raise NotImplementedError("calculate_ddf must be implemented")

    def calculate_wa(self) -> List[float]:
        """
        Compute weighted additive efficiency (WA) using data range-based weights.
        """
        # Combine input and output weights
        w_in = self._calculate_wa_w_inp()
        w_out = self._calculate_wa_w_out()
        # Implement weighted aggregation logic here
        raise NotImplementedError("calculate_wa must be implemented")

    def _calculate_wa_w_inp(self) -> List[float]:
        """
        Calculate weights for inputs inversely proportional to each variable's range.

        Returns:
            List[float]: Weights for each input variable.
        """
        df_in = self.data[self.inputs]
        ranges = df_in.max() - df_in.min()
        total_vars = len(self.inputs) + len(self.outputs)
        return (1 / (total_vars * ranges)).tolist()

    def _calculate_wa_w_out(self) -> List[float]:
        """
        Calculate weights for outputs inversely proportional to each variable's range.

        Returns:
            List[float]: Weights for each output variable.
        """
        df_out = self.data[self.outputs]
        ranges = df_out.max() - df_out.min()
        total_vars = len(self.inputs) + len(self.outputs)
        return (1 / (total_vars * ranges)).tolist()

    def calculate_rui(self) -> List[float]:
        """
        Compute return to unobserved inputs (RUI) efficiency.
        """
        raise NotImplementedError("calculate_rui must be implemented")

    def calculate_ruo(self) -> List[float]:
        """
        Compute return to unobserved outputs (RUO) efficiency.
        """
        raise NotImplementedError("calculate_ruo must be implemented")

    def calculate_erg(self) -> List[float]:
        """
        Compute global radial efficiency (ERG).
        """
        raise NotImplementedError("calculate_erg must be implemented")

    def calculate_cost_model(self, eps: float = 0.0) -> List[float]:
        """
        Compute cost model efficiency C(y,w) with optional epsilon.
        """
        raise NotImplementedError("calculate_cost_model must be implemented")

    def calculate_profit_model(self, eps: float = 0.0) -> List[float]:
        """
        Compute profit model efficiency P(w,p) with optional epsilon.
        """
        raise NotImplementedError("calculate_profit_model must be implemented")
