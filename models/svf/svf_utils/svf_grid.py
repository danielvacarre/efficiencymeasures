from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from itertools import product
from numpy import arange
from pandas import DataFrame
from tqdm import tqdm

from models.svf.svf_utils.grid import GRID


def search_contiguous_cell(cell: tuple) -> list:
    """
    Returns a list of contiguous cells (neighbors) for a given grid cell.

    Args:
        cell (tuple): Coordinates of the current cell in the grid.

    Returns:
        list: List of tuples representing neighboring cells in each dimension.
    """
    contiguous = []
    base = list(cell)
    for dim_index in range(len(base)):
        neighbor = base.copy()
        neighbor[dim_index] = base[dim_index] - 1
        if neighbor[dim_index] >= 0:
            contiguous.append(tuple(neighbor))
    return contiguous


class SVFGrid(GRID):
    """
    SVFGrid class: extends GRID to build and process a grid for SVF and SSVF models.
    """

    def __init__(
        self,

    ):
        """
        Initializes the SVFGrid.

        """
        super().__init__()
        self.grid_properties = None  # DataFrame to store cell IDs, knot values, phi, and neighbors
        self.virtual_grid = None # DataFrame to store virtual grid points and properties

    def create_grid(self, data, inputs, outputs, d, parallel) -> None:
        """
        Constructs the grid layout based on input data and hyperparameter d.

        Populates:
            - grid_properties: DataFrame with 'id_cell', 'value', 'phi', 'c_cells'.
            - knot_list: List of knot arrays for each input dimension.
            - virtual_grid: DataFrame of virtual (knot) points.
            - data_grid: Original data annotated with cell positions and phi.
        """
        # Initialize the grid_properties DataFrame
        self.grid_properties = DataFrame(columns=["id_cell", "value", "phi", "c_cells"])

        # Filter input columns and determine dimensionality
        x = data[inputs]
        n_dim = x.shape[1]

        # Build knot list and indices for each dimension
        knot_list = []
        index_list = []
        for dim in range(n_dim):
            col_values = x.iloc[:, dim]
            min_val, max_val = col_values.min(), col_values.max()
            step = (max_val - min_val) / d
            knots = [min_val + i * step for i in range(d + 1)]
            knot_list.append(knots)
            index_list.append(arange(len(knots)))

        # Generate all combinations of cell indices and knot values
        self.grid_properties["id_cell"] = list(product(*index_list))
        self.grid_properties["value"] = list(product(*knot_list))
        self.knot_list = knot_list

        # Calculate grid properties and data annotations
        self.calculate_grid_properties(parallel, outputs)
        self.calculate_virtual_grid(inputs)
        self.calculate_data_grid(data, inputs, outputs, parallel)

    def calculate_virtual_grid(self, inputs) -> None:
        """
        Builds a DataFrame of all virtual grid points (knot values) for model evaluation.
        """
        values = self.grid_properties["value"].tolist()
        self.virtual_grid = DataFrame(values, columns= inputs)

    def calculate_dmu_phi(self, cell: tuple, outputs: list) -> list:
        """
        Computes the phi transformation (binary indicator) for a given cell.

        Args:
            cell (tuple): Coordinates of the target cell in the grid.

        Returns:
            list: List of phi vectors (one per output dimension), each a list of 0/1 indicators.
        """
        n_cells = len(self.grid_properties)
        n_dim = len(cell)
        phi_flat = []

        # For each grid point, check if all dimensions in 'id_cell' <= cell
        for idx in range(n_cells):
            id_coord = self.grid_properties.loc[idx, "id_cell"]
            indicator = 1
            for dim_idx in range(n_dim):
                if cell[dim_idx] < id_coord[dim_idx]:
                    indicator = 0
                    break
            phi_flat.append(indicator)

        # Repeat phi for each output variable
        return [phi_flat.copy() for _ in outputs]

    def process_cell(self, value: tuple, outputs: list) -> tuple:
        """
        Processes a single cell value: finds its grid cell, phi vector, and contiguous cells.

        Args:
            value (tuple): Knot values of the cell to process.

        Returns:
            tuple: (cell_id, phi, contiguous_cells) or (None, None, None) on error.
        """
        try:
            cell = self.search_dmu(list(value))
            phi = self.calculate_dmu_phi(cell, outputs)
            neighbors = search_contiguous_cell(cell)
            return cell, phi, neighbors
        except Exception as e:
            print(f"Error processing cell {value}: {e}")
            return None, None, None

    def calculate_grid_properties(self, parallel, outputs) -> None:
        """
        Populates the 'phi' and 'c_cells' columns in grid_properties using parallel processing.
        """
        futures = {}
        with ThreadPoolExecutor(max_workers=parallel or None) as executor:
            for value in self.grid_properties["value"]:
                futures[executor.submit(self.process_cell, value, outputs)] = value

            results = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating grid properties"):
                value = futures[future]
                cell, phi, neighbors = future.result()
                if cell is not None:
                    results[value] = (phi, neighbors)

        self.grid_properties["phi"] = self.grid_properties["value"].map(lambda v: results.get(v, (None,))[0])
        self.grid_properties["c_cells"] = self.grid_properties["value"].map(lambda v: results.get(v, (None, None))[1])

    def calculate_data_grid(self, data, inputs, outputs, parallel) -> None:
        """
        Annotates the original data with cell positions, phi vectors, and contiguous cells.
        """
        df = data.copy()
        df = df[inputs + outputs]
        x_values = df[inputs].values.tolist()

        pos_list = [None] * len(x_values)
        phi_list = [None] * len(x_values)
        neighbors_list = [None] * len(x_values)

        futures = {}
        with ThreadPoolExecutor(max_workers=parallel or None) as executor:
            for idx, row in enumerate(x_values):
                futures[executor.submit(self.process_cell, tuple(row), outputs)] = idx

            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating data grid"):
                idx = futures[future]
                cell, phi, neighbors = future.result()
                pos_list[idx] = cell
                phi_list[idx] = phi
                neighbors_list[idx] = neighbors

        self.data_grid = df
        self.data_grid["pos"] = pos_list
        self.data_grid["phi"] = phi_list
        self.data_grid["c_cells"] = neighbors_list
