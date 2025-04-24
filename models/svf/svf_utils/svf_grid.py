from itertools import product
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from numpy import arange
from pandas import DataFrame
from tqdm import tqdm
from models.svf.svf_utils.grid import GRID

def search_contiguous_cell(cell: tuple[int, ...]) -> list[tuple[int, ...]]:
    """
    Identify contiguous grid cells for a given cell index.

    Parameters
    ----------
    cell : tuple of int
        Index of the current grid cell.

    Returns
    -------
    List of tuples
        Neighboring cell indices that differ by -1 in one dimension.
    """
    neighbors: list[tuple[int, ...]] = []
    for dim, idx in enumerate(cell):
        if idx > 0:
            neighbor = list(cell)
            neighbor[dim] -= 1
            neighbors.append(tuple(neighbor))
    return neighbors


class SVFGrid(GRID):
    """
    Grid generator for Support Vector Frontier (SVF) estimators.

    Computes a multi-dimensional grid of knot values, and evaluates
    data transformations (phi) and contiguous cells for both the
    virtual grid and original observations.
    """

    def __init__(
        self,
        data: DataFrame,
        inputs: list[str],
        outputs: list[str],
        d: int,
        parallel: int
    ) -> None:
        """
        Initialize SVFGrid parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset containing input and output variables.
        inputs : list of str
            Names of input feature columns.
        outputs : list of str
            Names of output target columns.
        d : int
            Number of partitions per dimension (grid resolution).
        parallel : int
            Number of worker processes for parallel tasks.
        """
        super().__init__(data, inputs, outputs, d, parallel)
        self.knot_list: list[list[float]] = []
        self.grid_properties: DataFrame = DataFrame()
        self.virtual_grid: DataFrame
        self.data_grid: DataFrame

    def create_grid(self) -> None:
        """
        Build the grid of knot values and compute its properties.

        Generates `grid_properties` with columns:
            - id_cell: tuple of grid indices
            - value: tuple of knot coordinates
            - phi: list of phi vectors per output
            - c_cells: contiguous cell neighbors

        Also constructs `virtual_grid` and `data_grid`.
        """
        x = self.data[self.inputs]
        n_dim = x.shape[1]

        # Generate knot_list and indices
        mins = x.min().values
        maxs = x.max().values
        amplitudes = (maxs - mins) / self.d
        self.knot_list = [
            list(mins[i] + amplitudes[i] * arange(self.d + 1))
            for i in range(n_dim)
        ]
        index_lists = [list(range(self.d + 1)) for _ in range(n_dim)]

        # Create grid properties DataFrame
        cells = list(product(*index_lists))
        values = [tuple(self.knot_list[i][idx[i]] for i in range(n_dim)) for idx in cells]
        self.grid_properties = DataFrame({'id_cell': cells, 'value': values})

        # Parallel compute phi and contiguous cells using threads to avoid pickle issues
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            results = list(executor.map(self._process_cell_props, self.grid_properties['value']))

        # Unzip results into phi and contiguous cell lists
        phi_list, c_cells_list = zip(*results)
        self.grid_properties['phi'] = list(phi_list)
        self.grid_properties['c_cells'] = list(c_cells_list)

        # Build virtual grid DataFrame
        self.virtual_grid = DataFrame(
            list(self.grid_properties['value']), columns=self.inputs
        )

        # Compute data_grid by mapping each observation
        self._build_data_grid(x.values.tolist())


    def _process_cell_props(self, cell_value: tuple[float, ...]) -> tuple[list[list[int]], list[tuple[int, ...]]]:
        """
        Internal helper: compute id_cell, phi, and contiguous cells for a given grid value.

        Parameters
        ----------
        cell_value : tuple of float
            Coordinates of the grid cell.

        Returns
        -------
        phi_list, contiguous_cells
        """
        # Find index of cell in grid_properties
        cell_idx = self.grid_properties.index[self.grid_properties['value'] == cell_value][0]
        id_cell = self.grid_properties.at[cell_idx, 'id_cell']

        all_ids = list(self.grid_properties['id_cell'])
        phi = [int(all(id_cell[d] >= other[d] for d in range(len(id_cell)))) for other in all_ids]

        # Replicate phi per output
        phi_list = [phi.copy() for _ in self.outputs]

        c_cells = search_contiguous_cell(id_cell)
        return phi_list, c_cells

    def _build_data_grid(self, x_list: list[list[float]]) -> None:
        """
        Attach grid position, phi, and contiguous cells to original data.
        """
        n_obs = len(x_list)
        pos_list: list[tuple[int, ...]] = [None] * n_obs
        phi_list: list[list[int]] = [None] * n_obs
        c_cells_list: list[list[tuple[int, ...]]] = [None] * n_obs

        # Map each observation to grid cell
        for i, obs in enumerate(tqdm(x_list, desc="Mapping data to grid")):
            # Find cell where all obs dims >= cell thresholds
            for idx, thresholds in zip(self.grid_properties['id_cell'], self.grid_properties['value']):
                if all(obs[d] >= thresholds[d] for d in range(len(idx))):
                    pos_list[i] = idx
                    phi_list[i] = self.grid_properties.loc[self.grid_properties['id_cell'] == idx, 'phi'].values[0]
                    c_cells_list[i] = self.grid_properties.loc[self.grid_properties['id_cell'] == idx, 'c_cells'].values[0]
                    break

        # Build data_grid
        self.data_grid = self.data.copy()
        self.data_grid['pos'] = pos_list
        self.data_grid['phi'] = phi_list
        self.data_grid['c_cells'] = c_cells_list
