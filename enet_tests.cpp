#include "enet.h"
#include <iostream>
#include <cmath>
#include <omp.h>
#include <chrono>


void test_copy_input_x_data(py::array_t<double> input_x){
    size_t N = input_x.shape(0), D = input_x.shape(1);
    std::cout << "shape = (" << N << " " << D << "), size = " << input_x.size() << std::endl;

    omp_set_num_threads(1);
    // copy data 
    std::vector<double> output = copy_input_x_data(input_x);

    // print it
    std::cout<<"printing output flattened array, fortran-format (column major)"<<std::endl;
    for(size_t i=0; i<N*D; i++){
        std::cout<<output[i]<<" ";
    }
    std::cout<<std::endl;
}

void test_copy_input_x_data_omp_time(py::array_t<double> input_x, int num_threads){
    size_t N = input_x.shape(0), D = input_x.shape(1);
    std::cout << "shape = (" << N << " " << D << "), size = " << input_x.size() << std::endl;

    omp_set_num_threads(num_threads);
    // copy data, and time it
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<double> output = copy_input_x_data(input_x);
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

// input_x should be of size N*D, means and stds of size D
void test_compute_mean_std_and_standardize_x_data(py::array_t<double> input_x, py::array_t<double> means, py::array_t<double> stds){
    size_t D = means.size();
    size_t N = input_x.size()/D;
    std::cout<<"input_x size= "<<input_x.size()<<", N="<<N<<", D="<<D<<std::endl;

    std::vector<double> data = copy_input_x_data(input_x);
    std::cout<<"printing means and stds of each column before standardization"<<std::endl;
    for(size_t j=0; j<D; j++){
        double mean=0, meansq=0;
        int n = 1;
        for(size_t i=j*N; i<(j+1)*(N); i++){
            mean += (data[i]-mean)/n;
            meansq += (data[i]*data[i]-meansq)/n;
            n++;
        }
        std::cout << "j="<< j << ": mean=" << mean << ", std=" << std::sqrt(meansq-mean*mean) << std::endl;
    }
    compute_mean_std_and_standardize_x_data(data, means, stds);
    // manually compute the mean and stdev of each column of the standardized data
    std::cout<<"printing means and stds of each column after standardization (should be 0 and 1)"<<std::endl;
    for(size_t j=0; j<D; j++){
        double mean=0, meansq=0;
        int n = 1;
        for(size_t i=j*N; i<(j+1)*(N); i++){
            mean += (data[i]-mean)/n;
            meansq += (data[i]*data[i]-meansq)/n;
            n++;
        }
        std::cout << "j="<< j << ": mean=" << mean << ", std=" << std::sqrt(meansq-mean*mean) << std::endl;
    }
}

// speed comparison for the standardization
void test_compute_mean_std_and_standardize_x_data_time(py::array_t<double> input_x, int D){
    py::array_t<double> means = py::array_t<double>(D), stds = py::array_t<double>(D);
    std::vector<double> data = copy_input_x_data(input_x);
    // time single thread performance
    std::vector<double> data1(data);
    omp_set_num_threads(1);
    auto start_time = std::chrono::high_resolution_clock::now();
    compute_mean_std_and_standardize_x_data(data1, means, stds);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout<< duration.count() << " seconds requried to standardize using 1 thread" << std::endl;
    // multithread performance
    std::vector<double> data2(data);
    omp_set_num_threads(4);
    start_time = std::chrono::high_resolution_clock::now();
    compute_mean_std_and_standardize_x_data(data2, means, stds);
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    std::cout<< duration.count() << " seconds requried to standardize using 4 threads" << std::endl;
}



PYBIND11_MODULE(enet_tests, m){
    m.doc() = "tests for elastic net";
    m.def("test_copy_input_x_data", &test_copy_input_x_data, "function to test copying data works");
    m.def("test_copy_input_x_data_omp_time", &test_copy_input_x_data_omp_time, "function for timing parallel data copy");
    m.def("test_compute_mean_std_and_standardize_x_data", &test_compute_mean_std_and_standardize_x_data, 
        "function to test standardizing data and computing per-column means and stds");
    m.def("test_compute_mean_std_and_standardize_x_data_time", &test_compute_mean_std_and_standardize_x_data_time, 
        "function to test the speed of standardizing data and computing per-column means and stds");
}
