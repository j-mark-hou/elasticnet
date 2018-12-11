// // TODO: at some point, write some speed comparison tests for the C++ code

// #include <iostream>
// #include <cmath>
// #include <omp.h>
// #include "enet_helpers.h"

// void test_copy_input_x_data_omp_time(const py::array_t<double> input_x, py::array_t<double> output_x, int num_threads){
//     size_t N = input_x.shape(0), D = input_x.shape(1);
//     std::cout << "shape = (" << N << " " << D << "), size = " << input_x.size() << std::endl;
//     // copy data, and time it
//     double start_time = omp_get_wtime();
//     copy_input_x_data(input_x, output_x, num_threads);
//     std::cout << "copying using " << num_threads << " threads took "<< omp_get_wtime()-start_time << " seconds " << std::endl;
//     // print the sum to check that it's correct
//     double vsum = 0;
//     auto output_x_unchecked = output_x.unchecked<1>();
//     #pragma omp parallel for schedule(static) reduction(+:vsum)
//     for(size_t i=0; i<N*D; i++){
//         vsum += output_x_unchecked(i);
//     }
//     std::cout << "sum of entries for sanity checking: " << vsum << std::endl;
// }


// void test_compute_mean_std_and_standardize_x_data_time(py::array_t<double> input_x, int D, int num_threads){
//     py::array_t<double> means = py::array_t<double>(D), stds = py::array_t<double>(D);
//     // time single thread performance
//     double start_time = omp_get_wtime();
//     compute_mean_std_and_standardize_x_data(input_x, means, stds, num_threads);
//     std::cout << "standardizing and mean/std generation using " << num_threads 
//                 << " threads took "<< omp_get_wtime()-start_time << " seconds " << std::endl;
// }


// void test_estimate_squaredloss_naive_time(py::array_t<double> x_standardized, py::array_t<double> input_y,
//                                           py::array_t<double> params_init, py::array_t<double> params,
//                                           double lambda, double alpha, 
//                                           double tol, size_t max_coord_descent_rounds,
//                                           int num_threads){
//     // time single thread performance
//     double start_time = omp_get_wtime();
//     int num_rounds = estimate_squaredloss_naive(x_standardized, input_y,
//                                                 params_init, params,
//                                                 lambda, alpha, 
//                                                 tol, max_coord_descent_rounds,
//                                                 num_threads);
//     std::cout << "elastic net estimation with " << num_threads << " threads completed " << num_rounds << " rounds in "
//               << omp_get_wtime()-start_time << " seconds " << std::endl;
// }


// PYBIND11_MODULE(enet_helpers_tests, m){
//     m.doc() = "tests for elastic net";
//     m.def("test_copy_input_x_data_omp_time", &test_copy_input_x_data_omp_time, "function for timing parallel data copy");
//     m.def("test_compute_mean_std_and_standardize_x_data_time", &test_compute_mean_std_and_standardize_x_data_time, 
//         "function to test the speed of standardizing data and computing per-column means and stds");
//     m.def("test_estimate_squaredloss_naive_time", &test_estimate_squaredloss_naive_time, 
//         "function to test the speed of elastic net");

// }
