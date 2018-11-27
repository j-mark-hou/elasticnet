import numpy as np
import enet

a = np.array(range(12)).reshape((3,4), order='A')
print(a.flags)

print()
b = np.array(range(12)).reshape((3,4), order='F')
print(b.flags)

enet.standardize_input_data(a)

enet.standardize_input_data(b)

print()
print(a.flags)
print()
print(b.flags)