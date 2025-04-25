from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, List, Union, Tuple, Dict

from ortools.linear_solver import pywraplp
from tqdm import tqdm

from models.svf.svf_methods.svf import SVF
from models.svf.svf_utils.svf_grid import SVFGrid
from models.svf.svf_utils.svf_primal_solution import SVFPrimalSolution

FMT = "%d-%m-%Y %H:%M:%S"


class SSVF(SVF):
    """
    Simplified Support Vector Frontier (SSVF) estimator.

    Extends SVF and implements a simplified linear programming model
    solved using OR-Tools.
    """

    def __init__(
        self,
        inputs: Union[List[str], Any],
        outputs: Union[List[str], Any],
        data: Any,
        C: float,
        eps: float,
        d: Union[int, float],
        parallel: int = 1
    ) -> None:
        """
        Initialize the SSVF estimator configuration.

        Parameters
        ----------
        inputs : list[str] or numpy.ndarray
            Names or array of input features.
        outputs : list[str] or numpy.ndarray
            Names or array of output targets.
        data : pandas.DataFrame or numpy.ndarray
            Dataset containing inputs and outputs.
        C : float
            Regularization parameter.
        eps : float
            Epsilon for the epsilon-insensitive loss.
        d : int or float
            Parameter for grid partitioning or derivative order.
        parallel : int, optional
            Number of threads for parallel operations (default=1).
        """
        super().__init__("SSVF", inputs, outputs, data, C, eps, d, parallel)

    def calculate_restriction(
        self,
        obs: int,
        out: int,
        w_var: Dict[Tuple[int, int], pywraplp.Variable],
        y: List[List[float]]
    ) -> Tuple[pywraplp.LinearExpr, int, int]:
        """
        Build the linear expression for a constraint (without adding to solver).

        Parameters
        ----------
        obs : int
            Observation index.
        out : int
            Output index.
        w_var : dict
            Dict mapping (output, variable) keys to solver variables.
        y : list of lists
            Observed output values.

        Returns
        -------
        expr : pywraplp.LinearExpr
            Left-hand side expression of the constraint.
        obs : int
            Observation index (for sorting).
        out : int
            Output index (for sorting).
        """
        n_vars = len(self.grid.grid_properties.phi[0][0])
        expr = y[obs][out]
        for var_idx in range(n_vars):
            coeff = self.grid.data_grid.phi[obs][out][var_idx]
            expr -= w_var[(out, var_idx)] * coeff
        return expr, obs, out

    def train(self) -> None:
        """
        Build and solve the SSVF linear program using OR-Tools.

        Sets:
            - self.model: solved OR-Tools solver instance.
            - self.train_time: duration of training.
        """
        start = datetime.now()
        # Extract observed outputs
        y_df = self.data.filter(self.outputs)
        y = y_df.values.tolist()
        n_out = len(self.outputs)
        n_obs = len(y)

        # Build evaluation grid
        self.grid = SVFGrid()
        self.grid.create_grid(self.data, self.inputs, self.outputs, self.d, self.parallel)

        # Define variable names
        w_keys = [(out, var) for out in range(n_out)
                  for var in range(len(self.grid.grid_properties.phi[0][0]))]
        xi_keys = [(out, obs) for out in range(n_out)
                   for obs in range(n_obs)]

        # Initialize solver
        solver = pywraplp.Solver.CreateSolver('GLOP')
        try:
            solver.SetNumThreads(self.parallel)
        except Exception:
            pass

        # Create decision variables
        w_var = {key: solver.NumVar(0.0, solver.infinity(), f'w_{key[0]}_{key[1]}')
                 for key in w_keys}
        xi_var = {key: solver.NumVar(0.0, solver.infinity(), f'xi_{key[0]}_{key[1]}')
                  for key in xi_keys}

        # Objective: minimize C * sum(w) + eps * sum(xi)
        objective = solver.Objective()
        for key in w_keys:
            objective.SetCoefficient(w_var[key], self.C)
        for key in xi_keys:
            objective.SetCoefficient(xi_var[key], self.eps)
        objective.SetMinimization()

        # Build constraints in parallel
        restrictions = []
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = [executor.submit(self.calculate_restriction, obs, out, w_var, y)
                       for obs in range(n_obs) for out in range(n_out)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing constraints"):
                expr, obs, out = future.result()
                restrictions.append((expr, obs, out))

        # Add constraints sorted by observation and output
        for expr, obs, out in sorted(restrictions, key=lambda x: (x[1], x[2])):
            solver.Add(expr <= 0)
            solver.Add(-expr <= self.eps + xi_var[(out, obs)])

        # Solve LP
        solver.Solve()
        self.model = solver
        # Set derivative model if not provided
        if self.model_d is None:
            self.model_d = solver

        end = datetime.now()
        self.train_time = end - start

    def solve(self) -> None:
        """
        Extract solution values from the solved model and store in self.solution.

        Sets:
            - self.solution: SVFPrimalSolution containing w and xi matrices.
            - self.solve_time: duration of solution extraction.
        """
        start = datetime.now()
        vars_list = self.model.variables()

        sol_w: List[float] = []
        sol_xi: List[float] = []
        for var in vars_list:
            val = var.solution_value()
            if var.name().startswith('w_'):
                sol_w.append(val)
            else:
                sol_xi.append(val)

        # Organize solutions into matrices
        n_out = len(self.outputs)
        n_w_dim = len(sol_w) // n_out
        mat_w = [sol_w[i * n_w_dim:(i + 1) * n_w_dim] for i in range(n_out)]
        mat_xi = [sol_xi[i * len(self.data):(i + 1) * len(self.data)] for i in range(n_out)]

        self.solution = SVFPrimalSolution(mat_w, mat_xi)
        end = datetime.now()
        self.solve_time = end - start


