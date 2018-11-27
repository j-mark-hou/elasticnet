CC = g++
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` 
PYTHON3_SO_SUFFIX = `python3-config --extension-suffix`

make_and_test_everything: enet_tests$(PYTHON3_SO_SUFFIX)
	pytest -s test.py

enet_tests$(PYTHON3_SO_SUFFIX): 
	$(CC) $(CFLAGS) -o enet_tests$(PYTHON3_SO_SUFFIX) enet.cpp enet.h tests.cpp

enet:
	$(CC) $(CFLAGS) -o enet`python3-config --extension-suffix` enet.cpp enet.h

clean:
	rm *$(PYTHON3_SO_SUFFIX)

.PHONY: make_and_test_everything clean