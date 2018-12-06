import enet_helpers as eh
import numpy as np

def test_init():
    print()
    N,D = 3,4
    x = np.random.uniform(size=(N,D))
    y = np.random.uniform(size=N)
    print("x, y, before turning into eh.Data")
    print(x)
    print(y)
    data = eh.Data(x=x, y=y)
    print("x, y, after turning into eh.Data.  x is in column-major format")
    print(data.get_x())
    print(data.get_y())
    print("N,D = {}".format((data.N, data.D)))
    print("means and stds of the columns of x")
    print(data.get_means())
    print(data.get_stds())