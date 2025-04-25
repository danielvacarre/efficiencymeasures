from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Union, Tuple, Dict
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class SVF:
    """
    Support Vector Frontier (SVF) estimator for production frontier estimation
    based on Valero-Carreras et al. (2022).

    Attributes
    ----------
    method : str
        Name of the estimation method to use (e.g., 'SSVF', 'SVF-Splines').
    inputs : List[str] or numpy.ndarray
        Names or array of input variables/features.
    outputs : List[str] or numpy.ndarray
        Names or array of output variables/targets.
    data : pandas.DataFrame or numpy.ndarray
        Dataset containing the inputs and outputs for estimation.
    C : float
        Regularization parameter controlling the trade-off between
        model complexity and training error.
    eps : float
        Epsilon value for insensitive loss.
    d : int or float
        Parameter specifying the partition of the grid.
    parallel : bool
        Flag indicating whether to enable parallel computation.
    grid : Any
        Grid object used for evaluation (e.g., SVFGrid instance).
    model : Any
        Trained primary model instance (e.g., SVR).
    model_d : Any
        Trained derivative/auxiliary model for grid partitions.
    solution : dict
        Solution details of the estimation problem.
    name : str, optional
        Optional identifier for the estimator instance.
    """

    def __init__(
        self,
        method: str,
        inputs: Union[List[str], np.ndarray],
        outputs: Union[List[str], np.ndarray],
        data: Union[pd.DataFrame, np.ndarray],
        C: float,
        eps: float,
        d: Union[int, float],
        parallel: bool = False
    ) -> None:
        """
        Initialize the SVF estimator with the specified configuration.

        Parameters
        ----------
        method : str
            Name of the estimation method to use (e.g., 'SSVF', 'SVF-Splines').
        inputs : List[str] or numpy.ndarray
            Names or array of input variables/features.
        outputs : List[str] or numpy.ndarray
            Names or array of output variables/targets.
        data : pandas.DataFrame or numpy.ndarray
            Dataset containing the inputs and outputs for estimation.
        C : float
            Regularization parameter controlling the trade-off between
            model complexity and training error.
        eps : float
            Epsilon value for insensitive loss.
        d : int or float
            Parameter specifying the partition of the grid.
        parallel : bool, optional
            Flag indicating whether to enable parallel computation (default=False).
        """
        self.method: str = method
        self.inputs: Union[List[str], np.ndarray] = inputs
        self.outputs: Union[List[str], np.ndarray] = outputs
        self.data: Union[pd.DataFrame, np.ndarray] = data
        self.C: float = C
        self.eps: float = eps
        self.d: Union[int, float] = d
        self.parallel: bool = parallel

        # Attributes defined during estimation
        self.grid: Any = None       # Evaluation grid points
        self.model: Any = None      # Primary trained model
        self.model_d: Any = None    # Derivative/auxiliary model
        self.solution: dict = {}    # Solution details
        self.name: Union[str, None] = None  # Optional identifier
        self._phi_map: dict = {}    # Cache for grid phi vectors

    def get_estimation(self, dmu: List[float]) -> List[float]:
        """
        Efficiently estimate outputs for a given Decision Making Unit (DMU).

        Parameters
        ----------
        dmu : list of float
            Input values for the DMU.

        Returns
        -------
        list of float
            Estimated output values for each output dimension, rounded to three decimals.

        Raises
        ------
        ValueError
            If the length of dmu does not match number of inputs.
        """
        # Validate input length
        if len(dmu) != len(self.inputs):
            raise ValueError(
                "The number of inputs of the DMU does not match the problem inputs."
            )

        # Find cell for DMU
        dmu_cell = self.grid.search_dmu(dmu)

        # Early exit for out-of-bounds DMU
        if -1 in dmu_cell:
            return [0.0] * len(self.outputs)

        # Build phi cache on first call
        if not self._phi_map:
            cells = self.grid.grid_properties['id_cell']
            phis = self.grid.grid_properties['phi']
            # Map each cell tuple to its phi list
            self._phi_map = {cell: phi for cell, phi in zip(cells, phis)}

        # Retrieve phi vector
        phi = self._phi_map.get(dmu_cell)
        if phi is None:
            # Fallback if not found
            return [0.0] * len(self.outputs)

        # Convert weights and phi to numpy arrays for vectorized dot
        w_arr = np.array(self.solution.w)              # shape (n_outputs, n_features)
        phi_arr = np.array(phi)                        # shape (n_outputs, n_features)

        # Compute element-wise product and sum across features
        est = (w_arr * phi_arr).sum(axis=1)

        # Round results
        return [round(float(val), 3) for val in est]

    def get_virtual_grid_estimation(self) -> pd.DataFrame:
        """
        Estimate outputs for all virtual grid points and update virtual_grid.

        This method:
        1. Ensures a solution is available by calling self.solve() if needed.
        2. Extracts input values as a NumPy array for performance.
        3. Defines an internal function to estimate a single row.
        4. Executes estimation in parallel if self.parallel > 1, otherwise single-threaded.
        5. Constructs a DataFrame of estimated outputs and concatenates it
           with the original input grid.

        """
        # Ensure the model solution is computed
        if self.solution is None:
            self.solve()

        # Convert virtual grid inputs to NumPy array for efficient iteration
        input_values = self.grid.virtual_grid[self.inputs].values

        # Internal estimator function for a single row
        def _estimate(row: np.ndarray) -> List[float]:
            # Convert row to list and call get_estimation
            return self.get_estimation(row.tolist())

        # Choose parallel or sequential execution
        if self.parallel and self.parallel > 1:
            # Determine a reasonable chunksize to balance overhead vs. throughput
            chunksize = max(1, len(input_values) // (self.parallel * 4))
            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                results = list(executor.map(_estimate, input_values, chunksize=chunksize))
        else:
            # Single-threaded fallback (no executor overhead)
            results = [_estimate(row) for row in input_values]

        # Build a DataFrame from the results, preserving the original index
        df_est = pd.DataFrame(
            results,
            columns=self.outputs,
            index=self.grid.virtual_grid.index
        )

        # Concatenate inputs and estimated outputs
        self.grid.virtual_grid = pd.concat(
            [self.grid.virtual_grid[self.inputs], df_est],
            axis=1
        )

    def plot_frontier(
            self,
            num_points: int = 100,
            show_data: bool = True
    ) -> go.Figure:
        """
        Plot the estimated production frontier (interactive), from 0 to max+1
        for each input, optionally overlaying original data.
        """
        n_inputs = len(self.inputs)
        n_outputs = len(self.outputs)
        if n_outputs != 1:
            raise ValueError("Solo soporta un output para visualización interactiva.")

        # Construir rangos automáticos: [0, max+1]
        ranges = {
            inp: (0.0, float(self.data[inp].max()) + 1.0)
            for inp in self.inputs
        }

        # 1D frontier
        if n_inputs == 1:
            inp = self.inputs[0]
            out = self.outputs[0]
            a, b = ranges[inp]
            xs = np.linspace(a, b, num_points)
            ys = [self.get_estimation([x])[0] for x in xs]

            # Figura line + scatter
            fig = px.line(x=xs, y=ys, labels={'x': inp, 'y': out}, title='Estimated frontier')
            if show_data:
                fig.add_scatter(x=self.data[inp], y=self.data[out],
                                mode='markers', marker=dict(color='red', size=6), name='Data')
            return fig

        # 2D frontier surface
        elif n_inputs == 2:
            x1, x2 = self.inputs
            out = self.outputs[0]
            a1, b1 = ranges[x1]
            a2, b2 = ranges[x2]
            xs1 = np.linspace(a1, b1, num_points)
            xs2 = np.linspace(a2, b2, num_points)
            X1, X2 = np.meshgrid(xs1, xs2)
            Z = np.zeros_like(X1)

            for i in range(num_points):
                for j in range(num_points):
                    Z[i, j] = self.get_estimation([X1[i, j], X2[i, j]])[0]

            # Crear superficie
            fig = go.Figure()
            fig.add_trace(go.Surface(
                x=xs1,
                y=xs2,
                z=Z,
                colorscale=[[0, 'blue'], [1, 'blue']],  # color uniforme
                showscale=False,
                name='Estimated Frontier'
            ))

            # Superponer datos originales
            if show_data:
                fig.add_trace(go.Scatter3d(
                    x=self.data[x1],
                    y=self.data[x2],
                    z=self.data[out],
                    mode='markers',
                    marker=dict(size=5, symbol='circle', color='red'),
                    name='Data',
                ))

            fig.update_layout(
                title='Estimated Frontier 3D',
                scene=dict(
                    xaxis_title=x1,
                    yaxis_title=x2,
                    zaxis_title=out
                )
            )
            return fig

        else:
            raise ValueError(
                'plot_frontier soporta solo 1 o 2 inputs con un único output.'
            )