import numpy as np
import enet_tests

def test_copy_data1():
    a = np.ascontiguousarray(np.array(range(12)).reshape((3,4)))
    print("array is {}".format("c-format" if a.flags['C_CONTIGUOUS'] else "fortran-format"))
    print(a)
    a2 = enet_tests.test_copy_data1(a)

    print()
    b = np.asfortranarray(np.array(range(12)).reshape((3,4)))
    print(b)
    print("array is {}".format("c-format" if b.flags['C_CONTIGUOUS'] else "fortran-format"))
    b2 = enet_tests.test_copy_data1(b)

def test_copy_data_omp_time():
    """
    testing how fast the copy process is using single core vs openmp.
    when the format is Fortran (so it's literally just a matter of copying the raw data), 
        multi-thread is slower, whereas when the format is C (and you need to re-order the data),
        multi-thread is faster.
    """
    print()
    N,D = 100000, 1000
    b = np.ones(shape=(N,D), order='F')
    print("testing copying for a fortran-format array (multithread should be slower here) ")
    enet_tests.test_copy_data_omp_time(b, 1)
    enet_tests.test_copy_data_omp_time(b, 4)


    a = np.ones(shape=(N,D), order='C')
    print("testing copying for a c-format array (multithread should be faster here) ")
    enet_tests.test_copy_data_omp_time(a, 1)
    enet_tests.test_copy_data_omp_time(a, 4)


# print(a)
# print("array is {}".format("C-style" if a.flags['C_CONTIGUOUS'] else "Fortran-style"))
# enet.standardize_input_data(a)
# print("array is {}".format("C-style" if a.flags['C_CONTIGUOUS'] else "Fortran-style"))

# print()
# print(b)
# print("array is {}".format("C-style" if b.flags['C_CONTIGUOUS'] else "Fortran-style"))
# enet.standardize_input_data(b)
# print("array is {}".format("C-style" if b.flags['C_CONTIGUOUS'] else "Fortran-style"))