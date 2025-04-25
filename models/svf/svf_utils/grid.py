from numpy import transpose


class GRID:
    """
    GRID class for the SVF model. A grid is a partition of the input space divided into cells.
    """

    def __init__(self):
        """
        GRID class constructor.

        """
        self.data_grid = None  # Will store the partitioned data grid
        self.knot_list = None  # List of grid knots for each dimension

    def search_dmu(self, dmu):
        """
        Returns the cell in which an observation is located within the grid.

        Args:
            dmu (list): Observation values to locate in the grid.

        Returns:
            tuple: Coordinates of the observation's position in the grid.
        """
        cell = []
        # Transpose knot_list for dimension-wise comparison
        r = transpose(self.knot_list)
        for dim_index in range(len(self.knot_list)):
            # Iterate through knots in this dimension
            for knot_index in range(len(self.knot_list[dim_index])):
                comparison = transformation(dmu[dim_index], r[knot_index][dim_index])
                if comparison < 0:
                    cell.append(knot_index - 1)
                    break
                if comparison == 0:
                    cell.append(knot_index)
                    break
                # If last knot and still greater, assign to last cell
                if comparison > 0 and knot_index == len(self.knot_list[dim_index]) - 1:
                    cell.append(knot_index)
                    break
        return tuple(cell)


def transformation(x_i: float, t_k: float) -> int:
    """
    Determines whether an observation value is less than, equal to, or greater than a grid knot.

    Args:
        x_i (float): Observation value to compare.
        t_k (float): Grid knot value to compare against.

    Returns:
        int: -1 if x_i < t_k, 0 if x_i == t_k, 1 if x_i > t_k.
    """

    if x_i < t_k:
        return -1
    elif x_i == t_k:
        return 0
    else:
        return 1
