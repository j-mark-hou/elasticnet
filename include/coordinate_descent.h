#ifndef COORDINATE_DESCENT_H_
#define COORDINATE_DESCENT_H_

#include "common.h"
#include "data.h"
#include <vector>


// // copies data from input_x to output_x
// // input_x is be an (N,D) numpy array in either c-style (row major) or fortran-style (column major)
// // output_x is a 1-dimensional numpy array of length N*D, designed to hold the entries of input_x 
// //  in fortran style (column-major)
// void copy_input_x_data(py::array_t<double> input_x, py::array_t<double> output_x, int num_threads);

// // takes x_data, which is a 1-dim numpy array representing a column-major 2-dim numpy array, 
// //  and then computes the mean and variance of each column and stores them into means and stds resp.
// // in addition, this function will standardize data, so that each column of x_data ends up being
// //  mean-0 and std-1.
// // means and stds should be numpy arrays with length equal to the number of columns
// void compute_mean_std_and_standardize_x_data(py::array_t<double> x_data,
//         py::array_t<double> means, py::array_t<double> stds, int num_threads);

// TODO: DOCUMENT THIS
int estimate_squaredloss_naive(Data& data,
                               py::array_t<double> params_init, py::array_t<double> params,
                               double lambda, double alpha, 
                               double tol, size_t max_coord_descent_rounds,
                               int num_threads);

#endif // COORDINATE_DESCENT_H_