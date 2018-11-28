#include "enet.h"
#include <iostream>
#include <omp.h>
#include <chrono>


void test_copy_data1(py::array_t<double> input){
    size_t N = input.shape(0), D = input.shape(1);
    std::cout << "shape = (" << N << " " << D << "), size = " << input.size() << std::endl;

    omp_set_num_threads(1);
    // copy data 
    std::vector<double> output = copy_input_data(input);

    // print it
    std::cout<<"printing output flattened array, fortran-format (column major)"<<std::endl;
    for(size_t i=0; i<N*D; i++){
        std::cout<<output[i]<<" ";
    }
    std::cout<<std::endl;
}

void test_copy_data_omp_time(py::array_t<double> input, int num_threads){
    size_t N = input.shape(0), D = input.shape(1);
    std::cout << "shape = (" << N << " " << D << "), size = " << input.size() << std::endl;

    omp_set_num_threads(num_threads);
    // copy data, and time it
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<double> output = copy_input_data(input);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout<< duration.count() << " seconds requried to copy using " <<  num_threads << " threads" << std::endl;

    // print the sum to check that it's correct
    double vsum = 0;
    #pragma omp parallel for schedule(static) reduction(+:vsum)
    for(size_t i=0; i<N*D; i++){
        vsum += output[i];
    }
    std::cout << "sum of entries for sanity checking: " << vsum << std::endl;
    // return vsum;
}


PYBIND11_MODULE(enet_tests, m){
    m.doc() = "tests for elastic net";
    m.def("test_copy_data1", &test_copy_data1, "function to test copying data works");
    m.def("test_copy_data_omp_time", &test_copy_data_omp_time, "function for timing parallel data copy");
}
