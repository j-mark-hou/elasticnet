import numpy as np
import enet_helpers
import enet_helpers_tests
import time

def test_copy_input_x_data():
    print()
    N, D = 3, 4
    for format_func in (np.ascontiguousarray, np.asfortranarray):
        input_arr = format_func(np.array(range(N*D)).reshape((N,D)))
        output_arr = np.empty(shape=(N*D))
        print("copying array from {} into F-format"
                .format("C-format" if input_arr.flags['C_CONTIGUOUS'] else "F-format"))
        enet_helpers.copy_input_x_data(input_arr, output_arr, 1)
        print(input_arr)
        print(output_arr)
    print("the copied 1-dim outputs should be identical even though inputs were different formats")

def test_copy_input_x_data_omp_time():
    """
    testing how fast the copy process is using single core vs openmp.
    when the format is Fortran (so it's literally just a matter of copying the raw data), 
        multi-thread is slower, whereas when the format is C (and you need to re-order the data),
        multi-thread is faster.
    """
    print()
    N,D = 100000, 1000
    for order in ['C', 'F']:
        print("TESTING COPY for a {}-format array".format(order))
        for num_threads in (1,4):
            input_arr = np.ones(shape=(N,D), order=order)
            output_arr = np.empty(shape=(N*D))
            enet_helpers_tests.test_copy_input_x_data_omp_time(input_arr, output_arr, num_threads)
    print("multithread faser for both C and F-format")

def test_compute_mean_std_and_standardize_x_data():
    print()
    N,D = 100, 5 # small data
    x_data = np.random.uniform(size=N*D) # mean is .5, std is sqrt(1/12)~=.288
    means, stds = np.empty(shape=D), np.empty(shape=D)
    enet_helpers.compute_mean_std_and_standardize_x_data(x_data, means, stds, 1)
    # print the output means and stds
    print("The computed means and stds should be around .5 and around .288 respectively")
    print("means", means)
    print("stds", stds)

def test_compute_mean_std_and_standardize_x_data_time():
    N,D = 100000, 1000
    print()
    print("data shape is {}".format((N,D)))
    for num_threads in (1,4):
        data_input = np.random.uniform(size=N*D)
        enet_helpers_tests.test_compute_mean_std_and_standardize_x_data_time(data_input, D, num_threads)

def test_estimate_squaredloss_naive1():
    # x-data: 
    N,D = 100, 6
    input_x = np.random.uniform(size=(N,D))
    x_standardized = np.empty(N*D)
    # copy it somewhere
    enet_helpers.copy_input_x_data(input_x, x_standardized, 1)
    # standardize it
    means = np.empty(D)
    stds = np.empty(D)
    enet_helpers.compute_mean_std_and_standardize_x_data(x_standardized, means, stds, 1)
    # comput y.  coefs are [0, -1, 2, 0, -4, 5, 0, -7, -8, ...]
    input_y = np.zeros(N)
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
        input_y += x_standardized[(j*N):((j+1)*N)] * coef
    # define parameters needed to run the enet estimationg
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
    num_rounds = enet_helpers.estimate_squaredloss_naive(x_standardized, input_y, 
                                                 params_init, params, 
                                                 reg_lambda, reg_alpha, 
                                                 tol, max_coord_descent_rounds,
                                                 num_threads)
    print("number of rounds (=passes through the coordinates): {}".format(num_rounds))
    print("computed means :  [{}]".format(", ".join(["{:.2f}".format(x) for x in means])))
    print("computed stds :   [{}]".format(", ".join(["{:.2f}".format(x) for x in stds])))
    print("initial params :  [{}]".format(", ".join(["{:.2f}".format(x) for x in params_init])))
    print("computed params : [{}]".format(", ".join(["{:.2f}".format(x) for x in params])))
    print("real params :     [{}]".format(", ".join(["{:.2f}".format(x) for x in true_params])))


def test_estimate_squaredloss_naive_time():
    # x-data: 
    N,D = 1000000, 100
    input_x = np.random.uniform(size=(N,D))
    x_standardized = np.empty(N*D)
    # copy it somewhere
    enet_helpers.copy_input_x_data(input_x, x_standardized, 1)
    # standardize it
    means = np.empty(D)
    stds = np.empty(D)
    enet_helpers.compute_mean_std_and_standardize_x_data(x_standardized, means, stds, 1)
    # comput y.  coefs are [0, -1, 2, 0, -4, 5, 0, -7, -8, ...]
    input_y = np.zeros(N)
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
        input_y += x_standardized[(j*N):((j+1)*N)] * coef
    # define parameters needed to run the enet estimationg
    params_init = np.zeros(shape=D)
    params = np.empty(shape=D)
    reg_lambda = .01
    reg_alpha = .5
    max_coord_descent_rounds = 100
    tol = .001
    print()
    print("N={}, D={}".format(N,D))
    print("real params :     [{}]".format(", ".join(["{:.2f}".format(x) for x in true_params])))
    # multithread shoul be like 2/3 the time.  this example doesn't go many rounds, so most
    #  of the speedup is from faster copying ofdata rather than faster iteration.
    for num_threads in (1,4):
        print("ESTIMATING USING {} THREADS".format(num_threads))
        print("l2 objective, lambda={}, alpha={}, tol={}, up to {} rounds, using {} threads"
                .format(reg_lambda, reg_alpha, tol, max_coord_descent_rounds, num_threads))
        enet_helpers_tests.test_estimate_squaredloss_naive_time(x_standardized, input_y, 
                                                        params_init, params, 
                                                        reg_lambda, reg_alpha, 
                                                        tol, max_coord_descent_rounds,
                                                        num_threads)
        print("computed params : [{}]".format(", ".join(["{:.2f}".format(x) for x in params])))
