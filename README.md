Python package for coordinate-descent based optimization for elastic net.

Example notebook: https://github.com/j-mark-hou/elasticnet/blob/master/example.ipynb

Some notes:

- computation is in C++
	- support multithreading via OpenMP
- interface is in Python (see `elasticnet.py`), which accesses the built c++ shared library via pybind
- currently only supports L2 objective (more on the way)

Building:

- run `make` from directory
	- the `makefile` is specific to my particular version of python, so it probably won't work in general
	- `makefile` is also very specific to linux, so won't work elsewhere
- run `example.ipynb` once `make` completes