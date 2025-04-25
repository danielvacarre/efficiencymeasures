from typing import List, Optional
from ortools.linear_solver import pywraplp
from models.efficiency.efficiency_method import EfficiencyMethod


class CSVFEff(EfficiencyMethod):
    """
    CSVF Efficiency Analysis using OR-Tools MIP solver.

    Supports:
      - Input-oriented BCC (BCC-RI)
      - Output-oriented BCC (BCC-RO)
      - Directional Distance Function (DDF)
      - Weighted Additive (WA)
      - Input Russell (RUI)
      - Output Russell (RUO)
      - Enhanced Russell Graph (ERG)
      - Cost model (C)
      - Profit model (P)
    """

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        data,
        df_estimation,
        weights_cols: Optional[List[str]] = None,
        prices_cols: Optional[List[str]] = None,
        eps: float = 0.0
    ):
        """
        Initialize the CSVFEff instance.

        Args:
            inputs: Names of input variable columns.
            outputs: Names of output variable columns.
            data: Original DataFrame.
            df_estimation: DataFrame used for frontier estimation.
            weights_cols: Column names for input weights (one per input).
            prices_cols: Column names for output prices (one per output).
            eps: Epsilon parameter.
        """
        super().__init__(inputs, outputs, data,
                         methods=[],
                         df_estimation=df_estimation,
                         weights_cols = weights_cols, prices_cols = prices_cols)

        # Store column names
        self.weights_cols = weights_cols or []
        self.prices_cols = prices_cols or []

        # Extract weight matrix from data, or default to 1s
        if self.weights_cols:
            self.weights = data[self.weights_cols].values.tolist()
        else:
            self.weights = [[1.0] * len(inputs) for _ in range(len(data))]

        # Extract price matrix from data, or default to 1s
        if self.prices_cols:
            self.prices = data[self.prices_cols].values.tolist()
        else:
            self.prices = [[1.0] * len(outputs) for _ in range(len(data))]

        # Convert to nested Python lists for OR-Tools
        self._inputs_data = data[inputs].values.tolist()
        self._outputs_data = data[outputs].values.tolist()
        self._inputs_est = df_estimation[inputs].values.tolist()
        self._outputs_est = df_estimation[outputs].values.tolist()

        self.num_observations = len(self._inputs_est)
        self.num_inputs = len(self.inputs)
        self.num_outputs = len(self.outputs)
        self.eps = eps

    def _new_solver(self, name: str) -> pywraplp.Solver:
        """
        Create a new OR-Tools MIP solver instance.

        Args:
            name: A descriptive name for the model.

        Returns:
            An OR-Tools MIP solver.
        """
        return pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')

    def calculate_ri(self, eps: float = None) -> List[float]:
        """
        Input-oriented BCC (variable returns to scale, BCC-RI).

        Minimize theta
        subject to:
            sum(lambda_k * x_est[k][i]) <= theta * x[o][i]
            sum(lambda_k * y_est[k][r]) >= y[o][r]
            sum(lambda_k) == 1
            lambda_k >= 0

        Returns:
            List of efficiency scores (theta) for each observation.
        """
        if eps is None:
            eps = self.eps
        efficiencies: List[float] = []
        for o in range(len(self.data)):
            solver = self._new_solver('BCC_RI')
            theta = solver.NumVar(0.0, solver.infinity(), 'theta')
            lam = [solver.IntVar(0.0, 1, f'lam_{k}')
                   for k in range(self.num_observations)]
            solver.Minimize(theta)

            # Input constraints
            for i in range(self.num_inputs):
                ct = solver.RowConstraint(-solver.infinity(), 0.0)
                ct.SetCoefficient(theta, -self._inputs_data[o][i])
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._inputs_est[k][i])

            # Output constraints
            for r in range(self.num_outputs):
                rhs = self._outputs_data[o][r]
                ct = solver.RowConstraint(rhs, solver.infinity())
                for k in range(self.num_observations):
                    coeff = self._outputs_est[k][r] + eps
                    ct.SetCoefficient(lam[k], coeff)

            # Convexity constraint
            ct_conv = solver.RowConstraint(1.0, 1.0)
            for k in range(self.num_observations):
                ct_conv.SetCoefficient(lam[k], 1.0)

            # Export model
            print(solver.ExportModelAsLpFormat(False))

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                efficiencies.append(round(theta.solution_value(), 6))
            else:
                efficiencies.append(0.0)
        return efficiencies

    def calculate_ro(self, eps: float = None) -> List[float]:
        """
        Output-oriented BCC (variable returns to scale, BCC-RO).

        Maximize phi
        subject to:
            sum(lambda_k * x_est[k][i]) <= x[o][i]
            sum(lambda_k * y_est[k][r]) >= phi * y[o][r]
            sum(lambda_k) == 1
            lambda_k >= 0

        Returns:
            List of efficiency scores (phi) for each observation.
        """
        if eps is None:
            eps = self.eps
        efficiencies: List[float] = []
        for o in range(len(self.data)):
            solver = self._new_solver('BCC_RO')
            phi = solver.NumVar(0.0, solver.infinity(), 'phi')
            lam = [solver.IntVar(0.0, 1, f'lam_{k}')
                   for k in range(self.num_observations)]
            solver.Maximize(phi)

            # Input constraints
            for i in range(self.num_inputs):
                ct = solver.RowConstraint(-solver.infinity(), self._inputs_data[o][i])
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._inputs_est[k][i])

            # Output constraints
            for r in range(self.num_outputs):
                ct = solver.RowConstraint(0.0, solver.infinity())
                ct.SetCoefficient(phi, -self._outputs_data[o][r])
                for k in range(self.num_observations):
                    coeff = self._outputs_est[k][r] + eps
                    ct.SetCoefficient(lam[k], coeff)

            # Convexity constraint
            ct_conv = solver.RowConstraint(1.0, 1.0)
            for k in range(self.num_observations):
                ct_conv.SetCoefficient(lam[k], 1.0)

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                efficiencies.append(round(phi.solution_value(), 6))
            else:
                efficiencies.append(0.0)
        return efficiencies

    def calculate_ddf(self, eps: float = None) -> List[float]:
        """
        Directional Distance Function (DDF).

        Maximize beta
        subject to:
            sum(lambda_k * x_est[k][i]) + beta * x[o][i] <= x[o][i]
            sum(lambda_k * y_est[k][r]) - beta * y[o][r] >= y[o][r]
            sum(lambda_k) == 1
            lambda_k >= 0

        Returns:
            List of efficiency scores (beta) for each observation.
        """
        if eps is None:
            eps = self.eps
        efficiencies: List[float] = []
        for o in range(len(self.data)):
            solver = self._new_solver('DDF')
            beta = solver.NumVar(0.0, solver.infinity(), 'beta')
            lam = [solver.IntVar(0.0, 1, f'lam_{k}')
                   for k in range(self.num_observations)]
            solver.Maximize(beta)

            # Input constraints
            for i in range(self.num_inputs):
                ct = solver.RowConstraint(-solver.infinity(), self._inputs_data[o][i])
                ct.SetCoefficient(beta, self._inputs_data[o][i])
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._inputs_est[k][i])

            # Output constraints
            for r in range(self.num_outputs):
                ct = solver.RowConstraint(self._outputs_data[o][r], solver.infinity())
                ct.SetCoefficient(beta, -self._outputs_data[o][r])
                for k in range(self.num_observations):
                    coeff = self._outputs_est[k][r] + eps
                    ct.SetCoefficient(lam[k], coeff)

            # Convexity constraint
            ct_conv = solver.RowConstraint(1.0, 1.0)
            for k in range(self.num_observations):
                ct_conv.SetCoefficient(lam[k], 1.0)

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                efficiencies.append(round(beta.solution_value(), 6))
            else:
                efficiencies.append(0.0)
        return efficiencies

    def calculate_wa(self, eps: float = None) -> List[float]:
        """
        Weighted Additive (WA).

        Maximize sum(s_neg[j] * w_in[j]) + sum(s_pos[r] * w_out[r])
        subject to:
            sum(lambda_k * x_est[k][j]) <= x[o][j] - s_neg[j]
            sum(lambda_k * y_est[k][r]) >= y[o][r] + s_pos[r]
            sum(lambda_k) == 1
            s_neg, s_pos, lambda_k >= 0

        Returns:
            List of aggregated slacks as efficiency scores.
        """
        if eps is None:
            eps = self.eps
        w_in = self._calculate_wa_w_inp()
        w_out = self._calculate_wa_w_out()
        efficiencies: List[float] = []
        for o in range(len(self.data)):
            solver = self._new_solver('WA')
            s_neg = [solver.NumVar(0.0, solver.infinity(), f's_neg_{j}') for j in range(self.num_inputs)]
            s_pos = [solver.NumVar(0.0, solver.infinity(), f's_pos_{r}') for r in range(self.num_outputs)]
            lam = [solver.IntVar(0.0, 1, f'lam_{k}')
                   for k in range(self.num_observations)]
            obj = solver.Objective()
            obj.SetMaximization()
            for j in range(self.num_inputs):
                obj.SetCoefficient(s_neg[j], w_in[j])
            for r in range(self.num_outputs):
                obj.SetCoefficient(s_pos[r], w_out[r])

            # Input constraints
            for j in range(self.num_inputs):
                ct = solver.RowConstraint(-solver.infinity(), self._inputs_data[o][j])
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._inputs_est[k][j])
                ct.SetCoefficient(s_neg[j], 1.0)

            # Output constraints
            for r in range(self.num_outputs):
                ct = solver.RowConstraint(self._outputs_data[o][r], solver.infinity())
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._outputs_est[k][r] + eps)
                ct.SetCoefficient(s_pos[r], -1.0)

            # Convexity constraint
            ct_conv = solver.RowConstraint(1.0, 1.0)
            for k in range(self.num_observations):
                ct_conv.SetCoefficient(lam[k], 1.0)

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                efficiencies.append(round(obj.Value(), 6))
            else:
                efficiencies.append(0.0)
        return efficiencies

    def calculate_rui(self, eps: float = None) -> List[float]:
        """
        Input Russell (RUI) BCC.

        Minimize sum(theta_j)
        subject to:
            sum(lambda_k * x_est[k][j]) <= theta_j * x[o][j]
            sum(lambda_k * y_est[k][r]) >= y[o][r]
            sum(lambda_k) == 1

        Efficiency = (sum of theta_j) / num_inputs.
        Returns list of efficiencies.
        """
        if eps is None:
            eps = self.eps
        efficiencies: List[float] = []
        for o in range(len(self.data)):
            solver = self._new_solver('RUI')
            theta = [solver.NumVar(0.0, solver.infinity(), f'theta_{j}') for j in range(self.num_inputs)]
            lam = [solver.IntVar(0.0, 1, f'lam_{k}')
                   for k in range(self.num_observations)]
            obj = solver.Objective()
            obj.SetMinimization()
            for t in theta:
                obj.SetCoefficient(t, 1.0)

            # Input constraints
            for j in range(self.num_inputs):
                ct = solver.RowConstraint(-solver.infinity(), 0.0)
                ct.SetCoefficient(theta[j], -self._inputs_data[o][j])
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._inputs_est[k][j])

            # Output constraints
            for r in range(self.num_outputs):
                ct = solver.RowConstraint(self._outputs_data[o][r], solver.infinity())
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._outputs_est[k][r] + eps)

            # Convexity constraint
            ct_conv = solver.RowConstraint(1.0, 1.0)
            for k in range(self.num_observations):
                ct_conv.SetCoefficient(lam[k], 1.0)

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                total = obj.Value()
                efficiencies.append(round(total / self.num_inputs, 6))
            else:
                efficiencies.append(0.0)
        return efficiencies

    def calculate_ruo(self, eps: float = None) -> List[float]:
        """
        Output Russell (RUO) BCC.

        Maximize sum(phi_r)
        subject to:
            sum(lambda_k * x_est[k][j]) <= x[o][j]
            sum(lambda_k * y_est[k][r]) >= phi_r * y[o][r]
            sum(lambda_k) == 1
            phi_r >= 1

        Efficiency = (sum of phi_r) / num_outputs.
        Returns list of efficiencies.
        """
        if eps is None:
            eps = self.eps
        efficiencies: List[float] = []
        for o in range(len(self.data)):
            solver = self._new_solver('RUO')
            phi = [solver.NumVar(1.0, solver.infinity(), f'phi_{r}') for r in range(self.num_outputs)]
            lam = [solver.IntVar(0.0, 1, f'lam_{k}')
                   for k in range(self.num_observations)]
            obj = solver.Objective()
            obj.SetMaximization()
            for p in phi:
                obj.SetCoefficient(p, 1.0)

            # Input constraints
            for j in range(self.num_inputs):
                ct = solver.RowConstraint(-solver.infinity(), self._inputs_data[o][j])
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._inputs_est[k][j])

            # Output constraints
            for r in range(self.num_outputs):
                ct = solver.RowConstraint(0.0, solver.infinity())
                ct.SetCoefficient(phi[r], -self._outputs_data[o][r])
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._outputs_est[k][r] + eps)

            # Convexity constraint
            ct_conv = solver.RowConstraint(1.0, 1.0)
            for k in range(self.num_observations):
                ct_conv.SetCoefficient(lam[k], 1.0)

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                total = obj.Value()
                efficiencies.append(round(total / self.num_outputs, 6))
            else:
                efficiencies.append(0.0)
        return efficiencies

    def calculate_erg(self, eps: float = None) -> List[float]:
        """
        Enhanced Russell Graph (ERG).

        Minimize beta - (1/num_inputs)*sum(t_neg_j / inputs[o][j])
        subject to:
            beta + (1/num_outputs)*sum(t_pos_r / outputs[o][r]) == 1
            -beta * inputs[o][j] + sum(lam[k] * inputs_est[k][j]) + t_neg[j] == 0
            -beta * outputs[o][r] + sum(lam[k] * outputs_est[k][r]) - t_pos[r] == 0
            sum(lam) == beta

        Returns:
            List of ERG efficiency scores (beta-based).
        """
        if eps is None:
            eps = self.eps
        efficiencies: List[float] = []
        for o in range(len(self.data)):
            solver = self._new_solver('ERG')
            beta = solver.NumVar(0.0, solver.infinity(), 'beta')
            t_neg = [solver.NumVar(0.0, solver.infinity(), f't_neg_{j}') for j in range(self.num_inputs)]
            t_pos = [solver.NumVar(0.0, solver.infinity(), f't_pos_{r}') for r in range(self.num_outputs)]
            lam = [solver.IntVar(0.0, 1, f'lam_{k}') for k in range(self.num_observations)]
            # Objective expression
            expr = beta
            for j in range(self.num_inputs):
                expr -= t_neg[j] / self._inputs_data[o][j] / self.num_inputs
            solver.Minimize(expr)
            # Constraint: beta + (1/num_outputs)*sum(t_pos / outputs[o]) == 1
            ct1 = solver.RowConstraint(1.0, 1.0)
            ct1.SetCoefficient(beta, 1.0)
            for r in range(self.num_outputs):
                ct1.SetCoefficient(t_pos[r], 1.0 / self._outputs_data[o][r] / self.num_outputs)
            # Constraints R2 & R3
            for j in range(self.num_inputs):
                c = solver.RowConstraint(0.0, 0.0)
                c.SetCoefficient(beta, -self._inputs_data[o][j])
                for k in range(self.num_observations): c.SetCoefficient(lam[k], self._inputs_est[k][j])
                c.SetCoefficient(t_neg[j], 1.0)
            for r in range(self.num_outputs):
                c = solver.RowConstraint(0.0, 0.0)
                c.SetCoefficient(beta, -self._outputs_data[o][r])
                for k in range(self.num_observations): c.SetCoefficient(lam[k], self._outputs_est[k][r] + eps)
                c.SetCoefficient(t_pos[r], -1.0)
            # Convexity: sum(lam) == beta
            c4 = solver.RowConstraint(0.0, 0.0)
            for k in range(self.num_observations): c4.SetCoefficient(lam[k], 1.0)
            c4.SetCoefficient(beta, -1.0)
            status = solver.Solve()
            efficiencies.append(round(solver.Objective().Value(), 6) if status == pywraplp.Solver.OPTIMAL else 0.0)
        return efficiencies

    def calculate_cost(self, eps: float = None) -> List[float]:
        """
        Cost model (C).

        Minimize sum(weights[o][j] * x_var[j])
        subject to:
            sum(lam[k] * inputs_est[k][j]) <= x_var[j]
            sum(lam[k] * outputs_est[k][r]) >= outputs[o][r]
            sum(lam) == 1

        Returns:
            List of cost-based efficiency scores.
        """
        if eps is None:
            eps = self.eps
        efficiencies: List[float] = []
        for o in range(len(self.data)):
            solver = self._new_solver('COST_MODEL')
            x_var = [solver.NumVar(0.0, solver.infinity(), f'x_{j}') for j in range(self.num_inputs)]
            lam = [solver.IntVar(0, 1, f'lam_{k}') for k in range(self.num_observations)]
            # Objective
            expr = solver.Sum(self.weights[o][j] * x_var[j] for j in range(self.num_inputs))
            solver.Minimize(expr)
            # Constraints
            for j in range(self.num_inputs):
                ct = solver.RowConstraint(-solver.infinity(), 0.0)
                for k in range(self.num_observations): ct.SetCoefficient(lam[k], self._inputs_est[k][j])
                ct.SetCoefficient(x_var[j], -1.0)
            for r in range(self.num_outputs):
                ct = solver.RowConstraint(self._outputs_data[o][r], solver.infinity())
                for k in range(self.num_observations): ct.SetCoefficient(lam[k], self._outputs_est[k][r] + eps)
            ct_conv = solver.RowConstraint(1.0, 1.0)
            for k in range(self.num_observations): ct_conv.SetCoefficient(lam[k], 1.0)
            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                efficiencies.append(round(expr.solution_value(), 6))
            else:
                efficiencies.append(0.0)
        return efficiencies

    def calculate_profit(self, eps: float = None) -> List[float]:
        """
        Profit model (P).

        Maximize: sum(prices[o][r] * y_var[r]) - sum(weights[o][j] * x_var[j])
        Subject to:
            sum(lambda_k * x_est[k][j]) <= x_var[j],  for each input j
            sum(lambda_k * (y_est[k][r] + eps)) >= y_var[r], for each output r
            sum(lambda_k) == 1
            lambda_k binary {0,1}
            x_var, y_var >= 0

        Returns:
            List of profit efficiency scores for each observation.
        """
        if eps is None:
            eps = self.eps
        efficiencies: List[float] = []

        for o in range(len(self.data)):
            solver = self._new_solver('PROFIT_MODEL')

            # Decision variables
            x_var = [solver.NumVar(0.0, solver.infinity(), f'x_{j}') for j in range(self.num_inputs)]
            y_var = [solver.NumVar(0.0, solver.infinity(), f'y_{r}') for r in range(self.num_outputs)]
            lam = [solver.IntVar(0.0, 1.0, f'lam_{k}') for k in range(self.num_observations)]

            # Objective: maximize revenue minus cost
            objective = solver.Objective()
            objective.SetMaximization()
            for r in range(self.num_outputs):
                objective.SetCoefficient(y_var[r], self.prices[o][r])
            for j in range(self.num_inputs):
                objective.SetCoefficient(x_var[j], -self.weights[o][j])

            # Input mixing constraints
            for j in range(self.num_inputs):
                ct = solver.RowConstraint(-solver.infinity(), 0.0)
                for k in range(self.num_observations):
                    ct.SetCoefficient(lam[k], self._inputs_est[k][j])
                ct.SetCoefficient(x_var[j], -1.0)

            # Output mixing constraints
            for r in range(self.num_outputs):
                ct = solver.RowConstraint(0.0, solver.infinity())
                for k in range(self.num_observations):
                    coeff = self._outputs_est[k][r] + eps
                    ct.SetCoefficient(lam[k], coeff)
                ct.SetCoefficient(y_var[r], -1.0)

            # Convexity constraint
            ct_conv = solver.RowConstraint(1.0, 1.0)
            for k in range(self.num_observations):
                ct_conv.SetCoefficient(lam[k], 1.0)

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                efficiencies.append(round(objective.Value(), 6))
            else:
                efficiencies.append(0.0)

        return efficiencies