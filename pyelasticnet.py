import numpy as np
import enet

def train(x, y, objective='l2', reg_lambdas=None, reg_alpha=.5, tol=.001, max_coord_descent_rounds=1000, num_threads=1):
    if objective=='l2':
        pass
    else:
        raise NotImplementedError("objective {} not available".format(objective))


class ElasticNetModel():
    pass