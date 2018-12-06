import elasticnet
import numpy as np

def test_init():
    print()
    N,D = 3,4
    x = np.random.uniform(size=(N,D))
    y = np.random.uniform(size=N)
    data = elasticnet.Data(x=x, y=y)
    print("x, after conversion")
    print(data.get_x())
    print(data.get_y())