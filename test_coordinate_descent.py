import enet_helpers as eh
import numpy as np

def test_cyclic_coordinate_descent():
    # x-data: 
    N,D = 100, 6
    x = np.random.uniform(size=(N,D))
    # compute y. intercept is 5, coefs are [0, -1, 2, 0, -4, 5, 0, -7, -8, ...]
    true_intercept = 5
    y = np.zeros(N) + true_intercept
    true_coefs = []
    for j in range(D):
        jmod3 = j%3
        if jmod3==0:
            coef = 0
        elif jmod3==1:
            coef = -j
        else:
            coef = j
        true_coefs.append(coef)
        y += (x[:,j]- x[:,j].mean())/(x[:,j].std())* coef
    # create the data
    data = eh.Data(x, y, num_threads=4)
    # define parameters needed to run the enet
    intercept = np.empty(shape=1)
    coefs = np.empty(shape=D)
    coefs_init = np.zeros(shape=D)
    reg_lambda = .01
    reg_alpha = .5
    max_coord_descent_rounds = 100
    tol = .001
    num_threads = 1
    print()
    print("l2 objective, lambda={}, alpha={}, tol={}, up to {} rounds, using {} threads"
            .format(reg_lambda, reg_alpha, tol, max_coord_descent_rounds, num_threads))
    num_rounds = eh.cyclic_coordinate_descent(data, "l2",
                                              intercept, coefs,
                                              coefs_init,
                                              reg_lambda, reg_alpha, 
                                              tol, max_coord_descent_rounds,
                                              num_threads)
    print("number of rounds (=passes through the coordinates): {}".format(num_rounds))
    print("computed means :   [{}]".format(", ".join(["{:.2f}".format(x) for x in data.get_means()])))
    print("computed stds :    [{}]".format(", ".join(["{:.2f}".format(x) for x in data.get_stds()])))
    print("initial coefs :    [{}]".format(", ".join(["{:.2f}".format(x) for x in coefs_init])))
    print("computed coefs :   [{}]".format(", ".join(["{:.2f}".format(x) for x in coefs])))
    print("computed intercept: {:.2f}".format(intercept[0]))
    print("true coefs :       [{}]".format(", ".join(["{:.2f}".format(x) for x in true_coefs])))
    print("true intercept:     {:.2f}".format(true_intercept))