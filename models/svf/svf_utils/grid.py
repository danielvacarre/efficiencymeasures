import pandas
from numpy import transpose


class GRID:
    """
    Clase GRID para realizar el modelo SVF. Un grid es una partición del espacio de los inputs que se divide en celdas.
    """

    def __init__(self, data: "pandas.DataFrame", inputs: list, outputs: list, d: list, parallel: int = 0):
        """
        Constructor de la clase GRID.

        Args:
            data (pandas.DataFrame): Conjunto de datos sobre los que se construye el grid.
            inputs (list): Listado de inputs.
            outputs (list): Listado de outputs.
            d (list): Número de particiones en las que se divide el grid.
            parallel: Número de procesadores a utilizar en el cálculo del grid.
        """
        self.data = data
        self.inputs = inputs
        self.outputs = outputs
        self.d = d
        self.parallel = parallel
        self.data_grid = None  # Aquí se almacenará la partición de datos en el grid
        self.knot_list = None  # Lista de nodos del grid para cada dimensión

    def search_dmu(self, dmu):
        """
            Función que devuelve la celda en la que se encuentra una observación en el grid
        Args:
            dmu (list): Observación a buscar en el grid
        Returns:
            position (list): Vector con la posición de la observación en el grid
        """
        cell = list()
        r = transpose(self.knot_list)
        for l in range(0, len(self.knot_list)):
            for m in range(0, len(self.knot_list[l])):
                trans = transformation(dmu[l], r[m][l])
                if trans < 0:
                    cell.append(m - 1)
                    break
                if trans == 0:
                    cell.append(m)
                    break
                if trans > 0 and m == len(self.knot_list[l]) - 1:
                    cell.append(m)
                    break
        return tuple(cell)

def transformation(x_i: float, t_k: float) -> int:
    """
    Evalúa si el valor de una observación es mayor, menor o igual a un nodo del grid.

    Args:
        x_i (float): Valor de la observación a evaluar.
        t_k (float): Valor del nodo con el que se compara.

    Returns:
        int: 1 si x_i > t_k, 0 si x_i == t_k, -1 si x_i < t_k.
    """

    if x_i < t_k:
        return -1
    elif x_i == t_k:
        return 0
    else:
        return 1