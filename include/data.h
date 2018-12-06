#ifndef DATA_H_
#define DATA_H_

#include "common.h"
#include <vector>

// base class for holding data
struct Data {
    std::vector<double> x; // x-data, size N*D, column-major format
    std::vector<double> y; // y-data, size N
    Data(py::array_t<double> x, py::array_t<double> y, int num_threads){
        omp_set_num_threads(num_threads);
        // copy the x
        auto x_unchecked = x.unchecked<2>();
        size_t N = x_unchecked.shape(0), D = x_unchecked.shape(1);
        this->x = std::vector<double>(N*D);
        // go through the array, column by column, filling things up in fortran order
        // (column-major)
        #pragma omp parallel for schedule(static) collapse(2)
        for(size_t j=0; j<D; j++){
            for(size_t i=0; i<N; i++){
                this->x[j*N+i] = x_unchecked(i,j);
            }
        }
        // copy the y
        auto y_unchecked = y.unchecked<1>();
        this->y = std::vector<double>(N);
        // go through the array, column by column, filling things up in fortran order
        // (column-major)
        #pragma omp parallel for schedule(static)
        for(size_t i=0; i<N; i++){
            this->y[i] = y_unchecked(i);
        }

    };
};


#endif