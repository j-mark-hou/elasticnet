CC = g++
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` 

enet:
	$(CC) $(CFLAGS) -o enet`python3-config --extension-suffix` enet.cpp

