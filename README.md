Python implementation coordinate-descent based optimization for elastic net.

For an example, see https://github.com/j-mark-hou/elasticnet/blob/master/example.ipynb

Some notes:

- computation is in c++ (see `enet_helpers.cpp`)
- interface is in python (see `elasticnet.py`), accesses c++ shared library via pybind
- currently only supports L2 objective (will add more as time goes on)

Building:

- run `make` from directory
	- the `makefile` is specific to my machine, so it may not work in general (will fix later)
	- `makefile` is also very linux-specific
- run `examply.ipynb` once `make` completes