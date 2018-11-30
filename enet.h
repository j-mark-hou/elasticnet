#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::vector<double> copy_input_x_data(py::array_t<double> input_X);

// takes data, which is a column-major 2-dim array, and then computes the mean and variance
//  of each column and stores them into means and stds, respectively.
// in addition, this function will standardize data, so that each column ends up being
//  mean-0 and std-1.
// means and stds should be numpy arrays with length equal to the number of columns
void compute_mean_std_and_standardize_x_data(std::vector<double>& x_data, py::array_t<double> means, py::array_t<double> stds);