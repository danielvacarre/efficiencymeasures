class SVFPrimalSolution:
    """
    Primal solution for the Support Vector Frontier (SVF) estimation.

    Attributes
    ----------
    w : list[list[float]]
        Weight vectors for each output dimension from the solved model.
    xi : list[list[float]]
        Slack variable values for each output dimension from the solved model.
    """

    def __init__(self, w: list[list[float]], xi: list[list[float]]) -> None:
        """
        Initialize the SVFPrimalSolution with computed model parameters.

        Parameters
        ----------
        w : list of list of float
            Solved weight values `w` for each output dimension.
        xi : list of list of float
            Solved slack variable values `xi` for each output dimension.
        """
        self.w = w
        self.xi = xi