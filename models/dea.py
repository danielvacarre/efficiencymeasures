from models.efficiency_method import EfficiencyMethod
from ortools.linear_solver import pywraplp

class DEA(EfficiencyMethod):
    """
    Data Envelopment Analysis (BCC-DEA) methods using OR-Tools MIP solver.

    Inherits from EfficiencyMethod and overrides calculation routines:
      - Input-oriented BCC (BCC-RI)
      - Output-oriented BCC (BCC-RO)
      - Directional Distance Function (DDF)
      - Weighted Additive (WA)
      - Input Russell BCC (RUI)
      - Output Russell BCC (RUO)
      - Enhanced Russell Graph (ERG)
    """

    def __init__(self, inputs, outputs, data, methods, df_estimation=None):
        super().__init__(inputs, outputs, data, methods, df_estimation)
        # Precompute numpy matrices and dimensions
        self.X = self.data[self.inputs].to_numpy()
        self.Y = self.data[self.outputs].to_numpy()
        self.n_obs, self.n_dim_x = self.X.shape
        _, self.n_dim_y = self.Y.shape

    def _new_solver(self, name: str):
        """Create a new OR-Tools MIP solver instance."""
        return pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')

    def calculate_ri(self) -> list[float]:
        """
        Input-oriented BCC (variable returns to scale, BCC-RI):
        minimize theta
        s.t. sum(lambda_k * x_kj) <= theta * x_oj
             sum(lambda_k * y_kr) >= y_or
             sum(lambda_k) == 1
             lambda_k >= 0
        """
        effs = []
        for o in range(self.n_obs):
            solver = self._new_solver('BCC-RI')
            theta = solver.NumVar(0.0, solver.infinity(), 'theta')
            lam = [solver.NumVar(0.0, solver.infinity(), f'lam_{k}') for k in range(self.n_obs)]

            solver.Minimize(theta)
            # Input constraints
            for j in range(self.n_dim_x):
                ct = solver.RowConstraint(-solver.infinity(), 0.0)
                ct.SetCoefficient(theta, -float(self.X[o, j]))
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.X[k, j]))
            # Output constraints
            for r in range(self.n_dim_y):
                ct = solver.RowConstraint(float(self.Y[o, r]), solver.infinity())
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.Y[k, r]))
            # Convexity constraint
            ct = solver.RowConstraint(1.0, 1.0)
            for k in range(self.n_obs):
                ct.SetCoefficient(lam[k], 1.0)

            # Export model
            # print(solver.ExportModelAsLpFormat(True))

            status = solver.Solve()
            effs.append(round(theta.solution_value(), 6) if status == pywraplp.Solver.OPTIMAL else 0.0)
        return effs

    def calculate_ro(self) -> list[float]:
        """
        Output-oriented BCC (variable returns to scale, BCC-RO):
        maximize phi
        s.t. sum(lambda_k * x_kj) <= x_oj
             sum(lambda_k * y_kr) >= phi * y_or
             sum(lambda_k) == 1
             lambda_k >= 0
        """
        effs = []
        for o in range(self.n_obs):
            solver = self._new_solver('BCC-RO')
            phi = solver.NumVar(0.0, solver.infinity(), 'phi')
            lam = [solver.NumVar(0.0, solver.infinity(), f'lam_{k}') for k in range(self.n_obs)]

            solver.Maximize(phi)
            # Input constraints
            for j in range(self.n_dim_x):
                ct = solver.RowConstraint(-solver.infinity(), float(self.X[o, j]))
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.X[k, j]))
            # Output constraints
            for r in range(self.n_dim_y):
                ct = solver.RowConstraint(0.0, solver.infinity())
                ct.SetCoefficient(phi, -float(self.Y[o, r]))
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.Y[k, r]))
            # Convexity constraint
            ct = solver.RowConstraint(1.0, 1.0)
            for k in range(self.n_obs):
                ct.SetCoefficient(lam[k], 1.0)

            # print(solver.ExportModelAsLpFormat(True))

            status = solver.Solve()
            effs.append(round(phi.solution_value(), 6) if status == pywraplp.Solver.OPTIMAL else 0.0)
        return effs

    def calculate_ddf(self) -> list[float]:
        """
        Directional Distance Function (DDF):
        maximize beta
        s.t. sum(lambda_k * x_kj) + beta * x_oj <= x_oj
             sum(lambda_k * y_kr) - beta * y_or >= y_or
             sum(lambda_k) == 1
             lambda_k >= 0
        """
        effs = []
        for o in range(self.n_obs):
            solver = self._new_solver('DDF')
            beta = solver.NumVar(0.0, solver.infinity(), 'beta')
            lam = [solver.NumVar(0.0, solver.infinity(), f'lam_{k}') for k in range(self.n_obs)]

            solver.Maximize(beta)
            # Input constraints
            for j in range(self.n_dim_x):
                ct = solver.RowConstraint(-solver.infinity(), float(self.X[o, j]))
                ct.SetCoefficient(beta, float(self.X[o, j]))
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.X[k, j]))
            # Output constraints
            for r in range(self.n_dim_y):
                ct = solver.RowConstraint(float(self.Y[o, r]), solver.infinity())
                ct.SetCoefficient(beta, -float(self.Y[o, r]))
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.Y[k, r]))
            # Convexity constraint
            ct = solver.RowConstraint(1.0, 1.0)
            for k in range(self.n_obs):
                ct.SetCoefficient(lam[k], 1.0)

            # print(solver.ExportModelAsLpFormat(True))

            status = solver.Solve()
            effs.append(round(beta.solution_value(), 6) if status == pywraplp.Solver.OPTIMAL else 0.0)
        return effs

    def calculate_wa(self) -> list[float]:
        """
        Weighted Additive (WA):
        maximize sum(s_neg_j * w_inp_j) + sum(s_pos_r * w_out_r)
        s.t. sum(lambda_k * x_kj) <= x_oj - s_neg_j
             sum(lambda_k * y_kr) >= y_or + s_pos_r
             sum(lambda_k) == 1
             s_neg_j, s_pos_r >= 0
             lambda_k >= 0
        """
        effs = []
        w_inp = self.calculate_wa_w_inp()
        w_out = self.calculate_wa_w_out()
        for o in range(self.n_obs):
            solver = self._new_solver('WA')
            s_neg = [solver.NumVar(0.0, solver.infinity(), f's_neg_{j}') for j in range(self.n_dim_x)]
            s_pos = [solver.NumVar(0.0, solver.infinity(), f's_pos_{r}') for r in range(self.n_dim_y)]
            lam = [solver.NumVar(0.0, solver.infinity(), f'lam_{k}') for k in range(self.n_obs)]

            obj = solver.Objective()
            obj.SetMaximization()
            for j, w in enumerate(w_inp):
                obj.SetCoefficient(s_neg[j], w)
            for r, w in enumerate(w_out):
                obj.SetCoefficient(s_pos[r], w)

            for j in range(self.n_dim_x):
                ct = solver.RowConstraint(-solver.infinity(), float(self.X[o, j]))
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.X[k, j]))
                ct.SetCoefficient(s_neg[j], 1.0)
            for r in range(self.n_dim_y):
                ct = solver.RowConstraint(float(self.Y[o, r]), solver.infinity())
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.Y[k, r]))
                ct.SetCoefficient(s_pos[r], -1.0)
            ct = solver.RowConstraint(1.0, 1.0)
            for k in range(self.n_obs):
                ct.SetCoefficient(lam[k], 1.0)

            # print(solver.ExportModelAsLpFormat(True))

            status = solver.Solve()
            val = obj.Value() if status == pywraplp.Solver.OPTIMAL else 0.0
            effs.append(round(val, 6))
        return effs

    def calculate_rui(self) -> list[float]:
        """
        Input Russell BCC (RUI):
        minimize sum(theta_j)
        s.t. sum(lambda_k * x_kj) <= theta_j * x_oj
             sum(lambda_k * y_kr) >= y_or
             sum(lambda_k) == 1
             theta_j, lambda_k >= 0
        efficiency = sum(theta_j) / n_dim_x
        """
        effs = []
        for o in range(self.n_obs):
            solver = self._new_solver('RUI')
            theta = [solver.NumVar(0.0, solver.infinity(), f'theta_{j}') for j in range(self.n_dim_x)]
            lam = [solver.NumVar(0.0, solver.infinity(), f'lam_{k}') for k in range(self.n_obs)]

            obj = solver.Objective()
            obj.SetMinimization()
            for t in theta:
                obj.SetCoefficient(t, 1.0)
            for j in range(self.n_dim_x):
                ct = solver.RowConstraint(-solver.infinity(), 0.0)
                ct.SetCoefficient(theta[j], -float(self.X[o, j]))
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.X[k, j]))
            for r in range(self.n_dim_y):
                ct = solver.RowConstraint(float(self.Y[o, r]), solver.infinity())
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.Y[k, r]))
            ct = solver.RowConstraint(1.0, 1.0)
            for k in range(self.n_obs):
                ct.SetCoefficient(lam[k], 1.0)

            # print(solver.ExportModelAsLpFormat(True))

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                total = obj.Value()
                effs.append(round(total / self.n_dim_x, 6))
            else:
                effs.append(0.0)
        return effs

    def calculate_ruo(self) -> list[float]:
        """
        Output Russell BCC (RUO):
        maximize sum(phi_r)
        s.t. sum(lambda_k * x_kj) <= x_oj
             sum(lambda_k * y_kr) >= phi_r * y_or
             sum(lambda_k) == 1
             phi_r >= 1, lambda_k >= 0
        efficiency = sum(phi_r) / n_dim_y
        """
        effs = []
        for o in range(self.n_obs):
            solver = self._new_solver('RUO')
            phi = [solver.NumVar(1.0, solver.infinity(), f'phi_{r}') for r in range(self.n_dim_y)]
            lam = [solver.NumVar(0.0, solver.infinity(), f'lam_{k}') for k in range(self.n_obs)]

            obj = solver.Objective()
            obj.SetMaximization()
            for p in phi:
                obj.SetCoefficient(p, 1.0)
            for j in range(self.n_dim_x):
                ct = solver.RowConstraint(-solver.infinity(), float(self.X[o, j]))
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.X[k, j]))
            for r in range(self.n_dim_y):
                ct = solver.RowConstraint(0.0, solver.infinity())
                ct.SetCoefficient(phi[r], -float(self.Y[o, r]))
                for k in range(self.n_obs):
                    ct.SetCoefficient(lam[k], float(self.Y[k, r]))
            ct = solver.RowConstraint(1.0, 1.0)
            for k in range(self.n_obs):
                ct.SetCoefficient(lam[k], 1.0)

            # print(solver.ExportModelAsLpFormat(True))

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                total = obj.Value()
                effs.append(round(total / self.n_dim_y, 6))
            else:
                effs.append(0.0)
        return effs

    def calculate_erg(self) -> list[float]:
        """
        Enhanced Russell Graph BCC (ERG):
        minimize beta - (1/n_dim_x) * sum(t_neg_j / x_oj)
        s.t. beta + (1/n_dim_y) * sum(t_pos_r / y_or) == 1
             -beta * x_oj + sum(lambda_k * x_kj) + t_neg_j == 0
             -beta * y_or + sum(lambda_k * y_kr) - t_pos_r == 0
             sum(lambda_k) == beta
             lambda_k, t_neg_j, t_pos_r >= 0
        """
        effs = []
        for o in range(self.n_obs):
            solver = self._new_solver('ERG')
            beta = solver.NumVar(0.0, solver.infinity(), 'beta')
            t_neg = [solver.NumVar(0.0, solver.infinity(), f't_neg_{j}') for j in range(self.n_dim_x)]
            t_pos = [solver.NumVar(0.0, solver.infinity(), f't_pos_{r}') for r in range(self.n_dim_y)]
            lam = [solver.NumVar(0.0, solver.infinity(), f'lam_{k}') for k in range(self.n_obs)]

            obj = solver.Objective()
            obj.SetMinimization()
            obj.SetCoefficient(beta, 1.0)
            for j in range(self.n_dim_x):
                obj.SetCoefficient(t_neg[j], -1.0 / (self.n_dim_x * float(self.X[o, j])))

            # R1: beta + (1/n_dim_y)*sum(t_pos/r) == 1
            ct1 = solver.RowConstraint(1.0, 1.0)
            ct1.SetCoefficient(beta, 1.0)
            for r in range(self.n_dim_y):
                ct1.SetCoefficient(t_pos[r], 1.0 / (self.n_dim_y * float(self.Y[o, r])))

            # R2 & R3
            for j in range(self.n_dim_x):
                c = solver.RowConstraint(0.0, 0.0)
                c.SetCoefficient(beta, -float(self.X[o, j]))
                for k in range(self.n_obs):
                    c.SetCoefficient(lam[k], float(self.X[k, j]))
                c.SetCoefficient(t_neg[j], 1.0)
            for r in range(self.n_dim_y):
                c = solver.RowConstraint(0.0, 0.0)
                c.SetCoefficient(beta, -float(self.Y[o, r]))
                for k in range(self.n_obs):
                    c.SetCoefficient(lam[k], float(self.Y[k, r]))
                c.SetCoefficient(t_pos[r], -1.0)

            # R4: sum(lam)=beta
            c4 = solver.RowConstraint(0.0, 0.0)
            for k in range(self.n_obs):
                c4.SetCoefficient(lam[k], 1.0)
            c4.SetCoefficient(beta, -1.0)

            # print(solver.ExportModelAsLpFormat(True))

            status = solver.Solve()
            effs.append(round(obj.Value(), 6) if status == pywraplp.Solver.OPTIMAL else 0.0)

        return effs
