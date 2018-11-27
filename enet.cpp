#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// input must be in column-major format,
void standardize_input_data(py::array_t<double, py::array::f_style> input){
    // auto input_unchecked = input.unchecked<2>();
    pybind11::buffer_info input_buf = input.request();
    double* input_ptr = (double*) input_buf.ptr;

    std::cout<<input.flags()<<std::endl;
    for(size_t i=0; i<input_buf.size; i++){
        std::cout<<input_ptr[i]<<" ";
    }
    std::cout<<std::endl;
}


PYBIND11_MODULE(enet, m){
    m.doc() = "elastic net";
    m.def("standardize_input_data", &standardize_input_data, 
            "function to standardize features (mean 0 var1) and convert to column-major format");
}
