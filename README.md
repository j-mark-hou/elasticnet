Python package for coordinate-descent based optimization for elastic net.

For an example, see https://github.com/j-mark-hou/elasticnet/blob/master/example.ipynb

Some notes:

- computation is in c++ (see `enet_helpers.cpp`)
	- support multithreading via OpenMP
- interface is in python (see `elasticnet.py`), which accesses the built c++ shared library via pybind
- currently only supports L2 objective (more on the way)

Building:

- run `make` from directory
	- the `makefile` is specific to my machine, so it probably won't work in general
	- `makefile` is also very linux-specific
- run `example.ipynb` once `make` completes