from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from ortools.linear_solver import pywraplp
from tqdm import tqdm

from models.svf.svf_methods.svf import SVF
from models.svf.svf_utils.svf_grid import SVFGrid
from models.svf.svf_utils.svf_primal_solution import SVFPrimalSolution

FMT = "%d-%m-%Y %H:%M:%S"


class SSVF(SVF):
    """
    Simplified Support Vector Frontier (SSVF) estimator.

    Inherits from SVF and implements a simplified optimization model using
    OR-Tools for solving the linear program.
    """

    def __init__(self, inputs, outputs, data, C, eps, d, parallel):
        """
        Initialize the SSVF estimator with the given configuration.

        Parameters
        ----------
        inputs : list[str] or numpy.ndarray
            Input feature names or array.
        outputs : list[str] or numpy.ndarray
            Output target names or array.
        data : pandas.DataFrame or numpy.ndarray
            Dataset containing inputs and outputs.
        C : float
            Regularization parameter.
        eps : float
            Epsilon for the epsilon-insensitive loss.
        d : int or float
            Parameter for grid dimensionality or derivative order.
        parallel : int
            Number of threads to use for parallel operations.
        """
        super().__init__("SSVF", inputs, outputs, data, C, eps, d, parallel)

    def calculate_restriction(self, obs, out, w_var, y):
        """
        Compute the linear expression for the restriction (without adding to solver).

        Returns a tuple of (expression, obs_index, out_index).
        """
        n_var = len(self.grid.grid_properties.phi[0][0])
        expr = y[obs][out]
        for var in range(n_var):
            coeff = self.grid.data_grid.phi[obs][out][var]
            expr -= w_var[(out, var)] * coeff
        return expr, obs, out

    def train(self):
        """
        Build the SSVF optimization problem and solve it using OR-Tools.

        Stores the solved solver in self.model and records training time.
        """
        start = datetime.now()
        y_df = self.data.filter(self.outputs)
        y = y_df.values.tolist()
        n_out = len(y_df.columns)
        n_obs = len(y)

        # Create evaluation grid
        self.grid = SVFGrid(self.data, self.inputs, self.outputs, self.d, self.parallel)
        self.grid.create_grid()
        print(self.grid.grid_properties)
        # Names for variables
        name_w = [(out, var) for out in range(n_out)
                  for var in range(len(self.grid.grid_properties.phi[0][0]))]
        print(name_w)
        name_xi = [(out, obs) for out in range(n_out)
                   for obs in range(n_obs)]
        print(name_xi)
        # Initialize OR-Tools solver (linear programming)
        solver = pywraplp.Solver.CreateSolver('GLOP')
        # Set number of threads if supported
        try:
            solver.SetNumThreads(self.parallel)
        except Exception:
            pass

        # Decision variables w and xi
        w_var = {key: solver.NumVar(0.0, solver.infinity(), f'w_{key[0]}_{key[1]}')
                 for key in name_w}
        xi_var = {key: solver.NumVar(0.0, solver.infinity(), f'xi_{key[0]}_{key[1]}')
                  for key in name_xi}

        # Objective: minimize sum(C*xi + w*initial_weight)
        objective = solver.Objective()
        for key in name_w:
            objective.SetCoefficient(w_var[key], self.C)
        for key in name_xi:
            objective.SetCoefficient(xi_var[key], self.eps)
        objective.SetMinimization()

        # Add constraints in parallel
        restrictions = []
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = [executor.submit(self.calculate_restriction, obs, out, w_var, y)
                       for obs in range(n_obs) for out in range(n_out)]
            for res in tqdm(as_completed(futures), total=len(futures), desc="Computing constraints"):
                restrictions.append(res.result())

        # Sort and add constraints to solver
        for expr, obs, out in sorted(restrictions, key=lambda x: (x[1], x[2])):
            solver.Add(expr <= 0)
            solver.Add(-expr <= self.eps + xi_var[(out, obs)])

        print(solver.ExportModelAsLpFormat(True))

        # Solve model
        solver.Solve()
        self.model = solver
        if self.model_d is None:
            self.model_d = solver

        end = datetime.now()
        self.train_time = end - start

    def solve(self):
        """
        Extract solution from the solved model and store it in self.solution.
        """
        start = datetime.now()
        vars_ = self.model.variables()
        # Separate w and xi values
        sol_w = []
        sol_xi = []
        for var in vars_:
            val = var.solution_value()
            if var.name().startswith('w_'):
                sol_w.append(val)
            else:
                sol_xi.append(val)

        n_out = len(self.outputs)
        n_w_dim = len(sol_w) // n_out
        mat_w = [sol_w[i*n_w_dim:(i+1)*n_w_dim] for i in range(n_out)]
        mat_xi = [sol_xi[i*len(self.data):(i+1)*len(self.data)] for i in range(n_out)]

        self.solution = SVFPrimalSolution(mat_w, mat_xi)
        end = datetime.now()
        self.solve_time = end - start
