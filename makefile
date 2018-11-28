CC = g++
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` 
# PYTHON3_SO_SUFFIX = `python3-config --extension-suffix` # somehow using this in place of the below doesn't work?

make_and_test: enet_tests.cpython-36m-x86_64-linux-gnu.so
	pytest -s test.py

enet_tests.cpython-36m-x86_64-linux-gnu.so: 
	$(CC) $(CFLAGS) -o enet_tests.cpython-36m-x86_64-linux-gnu.so enet.cpp enet.h tests.cpp

clean:
	rm *.so

.PHONY: make_and_test clean