import numpy as np
import enet
import enet_tests

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

    enet.estimate_squaredloss_naive()
