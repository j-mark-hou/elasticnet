#include <iostream>
#include <omp.h>
#include "enet.h"

// input must be in column-major format,
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


