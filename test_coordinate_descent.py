import enet_helpers as eh
import numpy as np

def test_estimate_squaredloss_naive1():
    # x-data: 
    N,D = 100, 6
    x = np.random.uniform(size=(N,D))
    # compute y.  coefs are [0, -1, 2, 0, -4, 5, 0, -7, -8, ...]
    y = np.zeros(N)
    true_params = []
    for j in range(D):
        jmod3 = j%3
        if jmod3==0:
            coef = 0
        elif jmod3==1:
            coef = -j
        else:
            coef = j
        true_params.append(coef)
        y += (x[:,j]- x[:,j].mean())/(x[:,j].std())* coef
    # create the data
    data = eh.Data(x, y, num_threads=4)
    # define parameters needed to run the enet
    params_init = np.zeros(shape=D)
    params = np.empty(shape=D)
    reg_lambda = .01
    reg_alpha = .5
    max_coord_descent_rounds = 100
    tol = .001
    num_threads = 1
    print()
    print("l2 objective, lambda={}, alpha={}, tol={}, up to {} rounds, using {} threads"
            .format(reg_lambda, reg_alpha, tol, max_coord_descent_rounds, num_threads))
    num_rounds = eh.estimate_squaredloss_naive(data, "l2",
                                             params_init, params, 
                                             reg_lambda, reg_alpha, 
                                             tol, max_coord_descent_rounds,
                                             num_threads)
    print("number of rounds (=passes through the coordinates): {}".format(num_rounds))
    print("computed means :  [{}]".format(", ".join(["{:.2f}".format(x) for x in data.get_means()])))
    print("computed stds :   [{}]".format(", ".join(["{:.2f}".format(x) for x in data.get_stds()])))
    print("initial params :  [{}]".format(", ".join(["{:.2f}".format(x) for x in params_init])))
    print("computed params : [{}]".format(", ".join(["{:.2f}".format(x) for x in params])))
    print("real params :     [{}]".format(", ".join(["{:.2f}".format(x) for x in true_params])))