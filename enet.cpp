#include <iostream>
#include <omp.h>
#include <cmath>
#include "enet.h"

// TODO: in the case that the input data is already fortran format, it would be probably faster to
//       just do `std::vector<double> output(input_ptr, input_ptr + N*D)`
//       with `(double *)input_ptr = (double *)input.request().ptr`
std::vector<double> copy_input_data(py::array_t<double> input){
    auto input_unchecked = input.unchecked<2>();
    size_t N = input_unchecked.shape(0), D = input_unchecked.shape(1);
    // initialize a new vector for copying over all the input data
    std::vector<double> output(N*D);
    // go through the array, column by column, filling things up in fortran order
    // (column-major)
    #pragma omp parallel for schedule(static) collapse(2)
    for(size_t c=0; c<D; c++){
        for(size_t r=0; r<N; r++){
            output[c*N+r] = input_unchecked(r,c);
        }
    }
    return output; // c++11 vector has move semantics so this won't result in excess copy
}


// data is a vector, representing a 2-d array in Fortran format 
//   as in, entries 1,...,N-1 correspond to column 0 of the array, etc.
// means, stds are numpy arrays of size D, so that the size of data is N*D
//   these two arrays exist to be populated with the means and stdevs of each column
// this function computes the means and stdevs of each of the D columns,
//   then takes the data, and then adjusts each column so that the mean and standard deviation
//   of each column are 0 and 1, respectively
void compute_mean_std_and_standardize_data(std::vector<double>& data, py::array_t<double> means, py::array_t<double> stds){
    // some data checks
    size_t D = means.size();
    size_t N = data.size()/D;
    if((D != stds.size()) || (N*D != data.size())){
        throw std::runtime_error("`means` and `stds` must both have size equal number of columns in `data`");
    }
    // get raw access to the numpy arrays
    auto means_unchecked = means.mutable_unchecked<1>();
    auto stds_unchecked = stds.mutable_unchecked<1>();
    // compute the means and stds for each dimension
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<D; j++){
        double mean=0, meansq=0;
        int n = 1;
        for(size_t i=j*N; i<(j+1)*(N); i++){
            mean += (data[i]-mean)/n;
            meansq += (data[i]*data[i]-meansq)/n;
            n++;
        }
        // store them in the relevant arrays
        means_unchecked[j] = mean;
        stds_unchecked[j] = std::sqrt(meansq-mean*mean);
    }
    // now normalize each column by subtracting the mean and dividing by the std
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<D; j++){
        for(size_t i=j*N; i<(j+1)*(N); i++){
            data[i] = (data[i]-means_unchecked(j))/stds_unchecked(j);
        }
    }
}


// PYBIND11_MODULE(enet, m){
//     m.doc() = "elastic net";
// }
