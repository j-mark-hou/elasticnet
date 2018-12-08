SRC_PATH = ./src/
INC_PATH = ./include/
TEST_PATH = ./tests/

# manually specify some target outputs.  get the relevant suffix by doing
#  `python3-config --extension-suffix`.  hardcoding it here for ease
MAIN_SO = enet_helpers.cpython-36m-x86_64-linux-gnu.so

CC = g++
CFLAGS = -O3 -Wall -shared -std=c++14 -fPIC -fopenmp -fvisibility=hidden `python3 -m pybind11 --includes` -I$(INC_PATH)

make_and_test: $(MAIN_SO) # $(TEST_SO)
	pytest -s test_data.py test_coordinate_descent.py

$(MAIN_SO): $(SRC_PATH)enet_helpers.cpp $(INC_PATH)data.h $(SRC_PATH)data.cpp \
			$(INC_PATH)coordinate_descent.h $(SRC_PATH)coordinate_descent.cpp \
			$(INC_PATH)objective.h
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm *.so

.PHONY: make_and_test clean