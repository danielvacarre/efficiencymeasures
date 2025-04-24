class SVF:
    """
    Support Vector Frontier (SVF) estimator for production frontier estimation
    based on Valero-Carreras et al. (2022).

    Attributes
    ----------
    method : str
        Name of the estimation method to use (e.g., 'SSVF', 'SVF-Splines').
    inputs : list[str] or numpy.ndarray
        Names or array of input variables/features.
    outputs : list[str] or numpy.ndarray
        Names or array of output variables/targets.
    data : pandas.DataFrame or numpy.ndarray
        Dataset containing the inputs and outputs for estimation.
    C : float
        Regularization parameter controlling the trade-off between
        model complexity and training error.
    eps : float
        Epsilon value.
    d : int or float
        Parameter specifying the partition of the grid
    parallel : bool
        Flag indicating whether to enable parallel computation.
    grid : any
        Grid from the model
    model : object
        Trained primary model instance (e.g., SVR).
    model_d : object
        Trained derivative model for d partitions of the grid.
    solution : dict
        Solution of the problem
    name : str
        Optional name identifier for the estimator instance.
    """
    def __init__(self, method, inputs, outputs, data, C, eps, d, parallel):
        """
        Initialize the SVF estimator with the specified configuration.

        Parameters
        ----------
        method : str
            Name of the estimation method to use (e.g., 'SSVF', 'SVF-Splines').
        inputs : list[str] or numpy.ndarray
            Names or array of input variables/features.
        outputs : list[str] or numpy.ndarray
            Names or array of output variables/targets.
        data : pandas.DataFrame or numpy.ndarray
            Dataset containing the inputs and outputs for estimation.
        C : float
            Regularization parameter controlling the trade-off between
            model complexity and training error.
        eps : float
            Epsilon value.
        d : int or float
            Parameter specifying the partition of the grid
        parallel : bool
            Flag indicating whether to enable parallel computation.
        """
        self.method = method
        self.inputs = inputs
        self.outputs = outputs
        self.data = data
        self.C = C
        self.eps = eps
        self.d = d
        self.parallel = parallel

        # To be defined during estimation
        self.grid = None       # Evaluation grid points
        self.model = None      # Primary trained model
        self.model_d = None    # Derivative/auxiliary model
        self.solution = None   # Solution details
        self.name = None       # Optional identifier for this estimator instance
