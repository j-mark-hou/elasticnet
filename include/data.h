#ifndef DATA_H_
#define DATA_H_

#include "common.h"
#include <vector>
#include <cmath>

// base class for holding data
class Data 
{
public:
    size_t N; // number of rows (= data points)
    size_t D; // number of columns (= features)
    std::vector<double> x; // x-data, size N*D, column-major format
    std::vector<double> y; // y-data, size N
    std::vector<double> means; // the means of teach column of x
    std::vector<double> stds; // and the standard devs
    Data(py::array_t<double> x, py::array_t<double> y, int num_threads=1);
private:
    // standardize the x data so that each column is mean-0 and std-1
    void compute_mean_std_and_standardize_x_data();
};


#endif