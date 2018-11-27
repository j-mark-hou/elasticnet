#include "enet.h"
#include <iostream>


void test_copy_data1(py::array_t<double> input){
    size_t N = input.shape(0), D = input.shape(1);
    std::cout << "shape " << N << " " << D << std::endl;
    std::cout << "size " << input.size() << std::endl;

    // copy data 
    std::vector<double> output = copy_input_data(input);

    // print it
    std::cout<<"printing output flattened array, fortran-format (column major)"<<std::endl;
    for(size_t i=0; i<N*D; i++){
        std::cout<<output[i]<<" ";
    }
    std::cout<<std::endl;
}



PYBIND11_MODULE(enet_tests, m){
    m.doc() = "tests for elastic net";
    m.def("test_copy_data1", &test_copy_data1, 
            "function to test copying data works");
}
