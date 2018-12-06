#ifndef DATA_H_
#define DATA_H_

#include "common.h"
#include <vector>

// base class for holding data
class Data {
public:
    size_t N; // number of rows (= data points)
    size_t D; // number of columns (= features)
    std::vector<double> x; // x-data, size N*D, column-major format
    std::vector<double> y; // y-data, size N
    std::vector<double> means; // the means of teach column of x
    std::vector<double> stds; // and the standard devs
    Data(py::array_t<double> x, py::array_t<double> y, int num_threads){
        omp_set_num_threads(num_threads);
        // copy the x
        auto x_unchecked = x.unchecked<2>();
        size_t N = x_unchecked.shape(0), D = x_unchecked.shape(1);
        this->N = N;
        this->D = D;
        this->x = std::vector<double>(N*D);
        // go through the array, column by column, filling things up in fortran order
        // (column-major)
        #pragma omp parallel for schedule(static) collapse(2)
        for(size_t j=0; j<D; j++){
            for(size_t i=0; i<N; i++){
                this->x[j*N+i] = x_unchecked(i,j);
            }
        }
        // standardize the data

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
// private:
//     // make x data mean-0 and std-1
//     void compute_mean_std_and_standardize_x_data(int num_threads){
//         omp_set_num_threads(num_threads);
//         size_t D = means.shape(0);
//         size_t N = y.shape(0);
//         size_t D = means.
//         // compute the means and stds for each dimension
//         #pragma omp parallel for schedule(static)
//         for(size_t j=0; j<D; j++){
//             double mean=0, meansq=0;
//             int n = 1;
//             for(size_t i=j*N; i<(j+1)*(N); i++){
//                 mean += (x_unchecked(i)-mean)/n;
//                 meansq += (x_unchecked(i)*x_unchecked(i)-meansq)/n;
//                 n++;
//             }
//             // store them in the relevant arrays
//             means_unchecked(j) = mean;
//             stds_unchecked(j) = std::sqrt(meansq-mean*mean);
//         }
//         // now normalize each column by subtracting the mean and dividing by the std
//         #pragma omp parallel for schedule(static)
//         for(size_t j=0; j<D; j++){
//             for(size_t i=j*N; i<(j+1)*(N); i++){
//                 x_unchecked(i) = (x_unchecked(i)-means_unchecked(j))/stds_unchecked(j);
//             }
//         }
//     };

};


#endif