#include <iostream>
#include <omp.h>
#include <cmath>
#include "enet.h"

// TODO: in the case that the input_x x_data is already fortran format, it would be probably faster to
//       just do `std::vector<double> output(input_x_ptr, input_x_ptr + N*D)`
//       with `(double *)input_x_ptr = (double *)input_x.request().ptr`
std::vector<double> copy_input_x_data(py::array_t<double> input_x){
    auto input_x_unchecked = input_x.unchecked<2>();
    size_t N = input_x_unchecked.shape(0), D = input_x_unchecked.shape(1);
    // initialize a new vector for copying over input_x
    std::vector<double> output(N*D);
    // go through the array, column by column, filling things up in fortran order
    // (column-major)
    #pragma omp parallel for schedule(static) collapse(2)
    for(size_t c=0; c<D; c++){
        for(size_t r=0; r<N; r++){
            output[c*N+r] = input_x_unchecked(r,c);
        }
    }
    return output; // c++11 vector has move semantics so this won't result in excess copy
}


// x_data is a vector, representing a 2-d array in Fortran format 
//   as in, entries 1,...,N-1 correspond to column 0 of the array, etc.
// means, stds are numpy arrays of size D, so that the size of x_data is N*D
//   these two arrays exist to be populated with the means and stdevs of each column
// this function computes the means and stdevs of each of the D columns,
//   then takes the x_data, and then adjusts each column so that the mean and standard deviation
//   of each column are 0 and 1, respectively
void compute_mean_std_and_standardize_x_data(std::vector<double>& x_data, py::array_t<double> means, py::array_t<double> stds){
    // some data checks
    size_t D = means.size();
    size_t N = x_data.size()/D;
    if((D != stds.size()) || (N*D != x_data.size())){
        throw std::runtime_error("`means` and `stds` must both have size equal number of columns in `x_data`");
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
            mean += (x_data[i]-mean)/n;
            meansq += (x_data[i]*x_data[i]-meansq)/n;
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
            x_data[i] = (x_data[i]-means_unchecked(j))/stds_unchecked(j);
        }
    }
}

// coordinate descent optimization for elasticnet with squared loss, using the 'naive' update strategy
//   (as opposed to the 'covariance' update strategy) 
// tol = when to terminate convergence if parameters don't change too much
// params_init = where to start estimation at
// params = array for holding the final estimated params
// max_coord_descent_rounds = how many rounds of coordinate descent (1 round = going through all coords once) to do at max
// lambda = the total regularization amount
// alpha = the fraction of regularization that goes on the L1 term (so 1-alpha goets on l2)
void estimate_squaredloss_naive(py::array_t<double> input_x, py::array_t<double> input_y,
                                py::array_t<double> means, py::array_t<double> stds,
                                py::array_t<double> params_init, py::array_t<double> params, size_t max_coord_descent_rounds,
                                double lambda, double alpha, double tol){
    // dimensionality of data
    size_t N = input_y.sie()
    size_t D = means.size();
    // TODO: add dimension check here... maybe?

    // compute the total l1 and l2 reg
    double l1_reg = alpha*lambda;
    double l2_reg = (1-alpha)*lambda;

    // initialize the final params at the value of the initial params
    auto params_init_unchecked = params_init.unchecked<1>();
    auto params_unchecked = params.mutable_unchecked<1>();
    for(size_t j=0; j<D; j++){
        params_unchecked(j) = params_init_unchecked(j);
    }
    // prepare x_data for estimation
    std::vector<double> x_data = copy_input_x_data(input_x);
    compute_mean_std_and_standardize_x_data(x_data, means, stds);
    // get a handle for accessing the y data
    y_unchecked = input_y.unchecked<1>();
    // compute the initial residuals (r in equation (7))
    // note that this number is never re-computed from first principles, rather
    // it's updated each time a param changes by subtracting the old and adding the new
    // TODO: do we need to worry about the value drifting further and further from truth?
    std::vector<double> resids(N);
    // initialize residuals at y
    for(size_t i=0; i<N; i++){
        resids[i] = y_unchecked[i];
    }
    // and then subtract the x_ij * params[j] repeatedly.
    for(size_t j=0; j<D; j++){
        // TODO: parallelize here
        for(size_t i=0; i<N; i++){
            resids[i] -= x_data[j*N+i] * params_unchecked[j]; //
        } 
    }
    // ok, now start estimating.
    double max_param_change; // the max chagne in the param value during a each round.
    double unregularized_optimal_param; // the value on `expression` inside the S(expression, \lambda\alpha) in equation (5)
    double tmp_param;
    for(size_t round=0; round<max_coord_descent_rounds; round++){
        max_param_change=0;
        // do one round of coordinate descent, where we iterate through all params and update each in turn
        for(size_t j=0; j<D; j++){
            // TODO: add the thing where we stop checking if a parameter becomes zero
            // compute the inside thing
            unregularized_optimal_param = params[j];
            for(size_t i=0; i<N; i++){
                unregularized_optimal_param += x_data[j*N+i] * resids[i] / N; // 
            }
            // see if it's big enough to warrant updating the params
            if(unregularized_optimal_param > l1_reg){
                //TODO: complete this
            }
        }
        // termination condition: params don't change enough
        if(max_param_change < tol){
            break;
        }
    }
    


}

PYBIND11_MODULE(enet, m){
    m.doc() = "elastic net";
    m.def("estimate_squaredloss_naive", &estimate_squaredloss_naive, 
        "coordinate descent optimization for elasticnet with squared loss, using the 'naive' update strategy");
}
