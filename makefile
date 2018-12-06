SRC_PATH = ./src/
INC_PATH = ./include/
TEST_PATH = ./tests/

# manually specify some target outputs.  get the relevant suffix by doing
#  `python3-config --extension-suffix`.  hardcoding it here for ease
MAIN_SO = enet_helpers.cpython-36m-x86_64-linux-gnu.so
TEST_SO = enet_helpers_tests.cpython-36m-x86_64-linux-gnu.so

CC = g++
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` -I$(INC_PATH)

make_and_test: $(MAIN_SO) # $(TEST_SO)
	pytest -s test_data.py

$(MAIN_SO): $(SRC_PATH)enet_helpers.cpp $(INC_PATH)data.h
	$(CC) $(CFLAGS) -o $(MAIN_SO) $(SRC_PATH)enet_helpers.cpp $(INC_PATH)data.h

# $(TEST_SO): $(SRC_PATH)enet_helpers.cpp $(INC_PATH)enet_helpers.h $(TEST_PATH)enet_helpers_tests.cpp
# 	$(CC) $(CFLAGS) -o $(TEST_SO) \
# 			$(SRC_PATH)enet_helpers.cpp $(INC_PATH)enet_helpers.h $(TEST_PATH)enet_helpers_tests.cpp

clean:
	rm *.so

.PHONY: make_and_test clean