from pandas import read_csv
from models.svf.svf_utils.inizialization import create_svf_method

if __name__ == "__main__":
    data = read_csv("C:/Users/dvc07/OneDrive/Escritorio/efficiencymeasures/data/prueba.txt", sep = ";", header = 0)

    inputs = ['x1','x2']
    outputs = ['y1', 'y2']
    C = 1
    eps = 0
    d = 10
    method = "SSVF"
    parallel = 2

    svf = create_svf_method(method, inputs, outputs, data, C, eps, d, parallel)
    svf.train()
    svf.solve()

    # Dibuja la frontera estimada en el rango observado de x
    # svf.plot_frontier(
    #     input_ranges={'x1': (0, 5)},
    #     num_points=1000
    # )

    svf.get_virtual_grid_estimation()

    # svf.plot_frontier(
    #     input_ranges={
    #         'x1': (0, 5),
    #     },
    #     num_points=10000,
    #     show_data=True
    # )