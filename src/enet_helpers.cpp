#include "data.h"
#include "coordinate_descent.h"

#include <iostream>
#include <omp.h>
#include <cmath>


PYBIND11_MODULE(enet_helpers, m){
    m.doc() = "elastic net";

    py::class_<Data>(m, "Data")
        .def(py::init<const py::array_t<double> &, const py::array_t<double> &, int>(),
            py::arg("x"), py::arg("y"), py::arg("num_threads")=1)
        // bind some lambda functions for returning the data.  Note that this will copy the internal
        //   data and return it, which may be quite costly
        .def_readonly("N", &Data::N)
        .def_readonly("D", &Data::D)
        .def("get_x", [](const Data &d) {
            return py::array_t<double>(d.x.size(), d.x.data());
        })
        .def("get_y", [](const Data &d) {
            return py::array_t<double>(d.y.size(), d.y.data());
        })
        .def("get_means", [](const Data &d) {
            return py::array_t<double>(d.means.size(), d.means.data());
        })
        .def("get_stds", [](const Data &d) {
            return py::array_t<double>(d.stds.size(), d.stds.data());
        });

    m.def("estimate_squaredloss_naive", &estimate_squaredloss_naive, 
        "coordinate descent optimization for elasticnet with squared loss, using the 'naive' update strategy");
}
