CC = g++
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -fopenmp
# PYTHON3_SO_SUFFIX = `python3-config --extension-suffix` # somehow using this in place of the below doesn't work?

make_and_test: enet.cpython-36m-x86_64-linux-gnu.so enet_tests.cpython-36m-x86_64-linux-gnu.so
	pytest -s test.py

enet.cpython-36m-x86_64-linux-gnu.so: enet.cpp enet.h
	$(CC) $(CFLAGS) -o enet.cpython-36m-x86_64-linux-gnu.so enet.cpp enet.h

enet_tests.cpython-36m-x86_64-linux-gnu.so: enet.cpp enet.h enet_tests.cpp
	$(CC) $(CFLAGS) -o enet_tests.cpython-36m-x86_64-linux-gnu.so enet.cpp enet.h enet_tests.cpp

clean:
	rm *.so

.PHONY: make_and_test clean