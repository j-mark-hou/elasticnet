import numpy as np
import enet
import enet_tests
import time

# def test_copy_input_x_data():
#     a = np.ascontiguousarray(np.array(range(12)).reshape((3,4)))
#     print("array is {}".format("c-format" if a.flags['C_CONTIGUOUS'] else "fortran-format"))
#     print(a)
#     a2 = enet_tests.test_copy_input_x_data(a)

#     print()
#     b = np.asfortranarray(np.array(range(12)).reshape((3,4)))
#     print(b)
#     print("array is {}".format("c-format" if b.flags['C_CONTIGUOUS'] else "fortran-format"))
#     b2 = enet_tests.test_copy_input_x_data(b)

# def test_copy_input_x_data_omp_time():
#     """
#     testing how fast the copy process is using single core vs openmp.
#     when the format is Fortran (so it's literally just a matter of copying the raw data), 
#         multi-thread is slower, whereas when the format is C (and you need to re-order the data),
#         multi-thread is faster.
#     """
#     print()
#     N,D = 100000, 1000
#     b = np.ones(shape=(N,D), order='F')
#     print("testing copying for a fortran-format array (multithread should be similar speed here) ")
#     enet_tests.test_copy_input_x_data_omp_time(b, 1)
#     enet_tests.test_copy_input_x_data_omp_time(b, 4)

#     a = np.ones(shape=(N,D), order='C')
#     print("testing copying for a c-format array (multithread should be faster here) ")
#     enet_tests.test_copy_input_x_data_omp_time(a, 1)
#     enet_tests.test_copy_input_x_data_omp_time(a, 4)

# def test_compute_mean_std_and_standardize_x_data():
#     print()
#     N,D = 100, 5 # small data
#     data_input = np.random.uniform(size=(N,D)) # mean is .5, std is sqrt(1/12)~=.288
#     means, stds = np.empty(shape=D), np.empty(shape=D)
#     enet_tests.test_compute_mean_std_and_standardize_x_data(data_input, means, stds)
#     # print the output means and stds
#     print("now, the computed means and stds (should be around .5 and around .288 respectively")
#     print("means", means)
#     print("stds", stds)

# def test_compute_mean_std_and_standardize_x_data_time():
#     N,D = 100000, 1000
#     data_input = np.random.uniform(size=(N,D))
#     print()
#     print("testing standardization speed using 1 core vs 4 cores")
#     print("data shape is {}, {}".format(N,D))
#     enet_tests.test_compute_mean_std_and_standardize_x_data_time(data_input, D)

def test_estimate_squaredloss_naive1():
    # simple array: 
    N,D = 100, 6
    input_x = np.random.uniform(size=(N,D))
    # input_x = np.random.normal(size=(N,D))
    # coefs are [0, -1, 2, 0, -4, 5, 0, -7, 8, ...]
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
        input_y += (input_x[:,j]-input_x[:,j].mean())/input_x[:,j].std() * coef
    means, stds = np.empty(shape=D), np.empty(shape=D)
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
    num_rounds = enet.estimate_squaredloss_naive(input_x, input_y, 
                                    means, stds, 
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


def test_estimate_squaredloss_time():
    N,D = 1000000, 100
    input_x = np.random.uniform(size=(N,D))
    # coefs are uniformly 1
    input_y = np.zeros(N)
    true_params = []
    for j in range(D):
        coef = j
        true_params.append(coef)
        input_y += (input_x[:,j]-input_x[:,j].mean())/input_x[:,j].std() * coef
    means, stds = np.empty(shape=D), np.empty(shape=D)
    params_init = np.zeros(shape=D)
    params = np.empty(shape=D)
    reg_lambda = .01
    reg_alpha = .5
    max_coord_descent_rounds = 1000
    tol = .001
    num_threads = 1
    print()
    print("N={}, D={}".format(N,D))
    print("real params :     [{}]".format(", ".join(["{:.2f}".format(x) for x in true_params])))
    # multithread shoul be like 2/3 the time
    for num_threads in [1, 4]:
        print("ESTIMATING USING {} THREADS".format(num_threads))
        print("l2 objective, lambda={}, alpha={}, tol={}, up to {} rounds, using {} threads"
                .format(reg_lambda, reg_alpha, tol, max_coord_descent_rounds, num_threads))
        t0 = time.time()
        num_rounds = enet.estimate_squaredloss_naive(input_x, input_y, 
                                                    means, stds, 
                                                    params_init, params, 
                                                    reg_lambda, reg_alpha, 
                                                    tol, max_coord_descent_rounds,
                                                    num_threads)
        t1 = time.time()
        print("computed params : [{}]".format(", ".join(["{:.2f}".format(x) for x in params])))
        print("{} threads, completed {} rounds in {:.3f} seconds".format(num_threads, num_rounds, t1-t0))
