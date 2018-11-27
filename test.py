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


# print(a)
# print("array is {}".format("C-style" if a.flags['C_CONTIGUOUS'] else "Fortran-style"))
# enet.standardize_input_data(a)
# print("array is {}".format("C-style" if a.flags['C_CONTIGUOUS'] else "Fortran-style"))

# print()
# print(b)
# print("array is {}".format("C-style" if b.flags['C_CONTIGUOUS'] else "Fortran-style"))
# enet.standardize_input_data(b)
# print("array is {}".format("C-style" if b.flags['C_CONTIGUOUS'] else "Fortran-style"))