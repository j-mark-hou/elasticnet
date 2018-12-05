#include <iostream>
#include <omp.h>
#include <cmath>
#include "enet_helpers.h"

#define DEBUG 0 // set to 1 to enable debug prints

// TODO: in the case that the input_x x_data is already fortran format, it would be probably faster to
//       just to... copy it
void copy_input_x_data(py::array_t<double> input_x, py::array_t<double> output_x, int num_threads){
    omp_set_num_threads(num_threads);
    auto input_x_unchecked = input_x.unchecked<2>();
    auto output_x_unchecked = output_x.mutable_unchecked<1>();
    size_t N = input_x_unchecked.shape(0), D = input_x_unchecked.shape(1);
    // go through the array, column by column, filling things up in fortran order
    // (column-major)
    #pragma omp parallel for schedule(static) collapse(2)
    for(size_t c=0; c<D; c++){
        for(size_t r=0; r<N; r++){
            output_x_unchecked(c*N+r) = input_x_unchecked(r,c);
        }
    }
}


// x_data is a 1-d numpy array, representing a 2-d array in Fortran format 
//   as in, entries 1,...,N-1 correspond to column 0 of the array, etc.
// means, stds are numpy arrays of size D, so that the size of x_data is N*D
//   these two arrays exist to be populated with the means and stdevs of each column
// this function computes the means and stdevs of each of the D columns,
//   then takes the x_data, and then adjusts each column so that the mean and standard deviation
//   of each column are 0 and 1, respectively
void compute_mean_std_and_standardize_x_data(py::array_t<double> x_data, 
        py::array_t<double> means, py::array_t<double> stds, int num_threads){
    omp_set_num_threads(num_threads);
    size_t D = means.shape(0);
    size_t N = x_data.shape(0)/D;
    // get raw access to the numpy arrays
    auto x_unchecked = x_data.mutable_unchecked<1>();
    auto means_unchecked = means.mutable_unchecked<1>();
    auto stds_unchecked = stds.mutable_unchecked<1>();
    // compute the means and stds for each dimension
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<D; j++){
        double mean=0, meansq=0;
        int n = 1;
        for(size_t i=j*N; i<(j+1)*(N); i++){
            mean += (x_unchecked(i)-mean)/n;
            meansq += (x_unchecked(i)*x_unchecked(i)-meansq)/n;
            n++;
        }
        // store them in the relevant arrays
        means_unchecked(j) = mean;
        stds_unchecked(j) = std::sqrt(meansq-mean*mean);
    }
    // now normalize each column by subtracting the mean and dividing by the std
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<D; j++){
        for(size_t i=j*N; i<(j+1)*(N); i++){
            x_unchecked(i) = (x_unchecked(i)-means_unchecked(j))/stds_unchecked(j);
        }
    }
}

// coordinate descent optimization for elasticnet with squared loss, using the 'naive' update strategy
//   (as opposed to the 'covariance' update strategy) 
// tol = when to terminate convergence if parameters don't change too much
// params_init = where to start estimation at
// params = array for holding the final estimated params.  no intercept returned because it's aways just the avg y 
// max_coord_descent_rounds = how many rounds of coordinate descent (1 round = going through all coords once) to do at max
// lambda = the total regularization amount
// alpha = the fraction of regularization that goes on the L1 term (so 1-alpha goets on l2)
int estimate_squaredloss_naive(py::array_t<double> x_standardized, py::array_t<double> input_y,
                               py::array_t<double> params_init, py::array_t<double> params,
                               double lambda, double alpha, 
                               double tol, size_t max_coord_descent_rounds,
                               int num_threads){
    omp_set_num_threads(num_threads);
    // dimensionality of data
    size_t N = input_y.size();
    size_t D = params.size();
    // TODO: add dimension check here... maybe?

    // compute the total l1 and l2 reg
    double l1_reg = alpha*lambda;
    double l2_reg = (1-alpha)*lambda;

    // initialize the final params at the value of the initial params
    auto params_init_unchecked = params_init.unchecked<1>();
    auto params_unchecked = params.mutable_unchecked<1>();
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<D; j++){
        params_unchecked(j) = params_init_unchecked(j);
    }

    // get access to input x-data, should be already standardized
    auto x_unchecked = x_standardized.unchecked<1>();
    // get access to y-data
    auto y_unchecked = input_y.unchecked<1>();

    // compute the initial residuals (r in equation (7)), which is (y - yhat)
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
        #pragma omp parallel for schedule(static) // can't parallelize above as all i's get updated for each j
        for(size_t i=0; i<N; i++){
            resids[i] -= x_unchecked(j*N+i) * params_unchecked[j]; //
        } 
    }
    // estimate by active-set iteration (see section 2.6), which amounts to these two steps:
    //  1. loop through all D params once
    //  2. then repeately loop through just the active params until no change
    //    - coords are removed from the active set as they become zero
    //  3. repeat
    //  4. terminate when a loop through all coefs fails
    std::vector<bool> inactive_params(D); // see 2.6
    bool current_round_is_for_only_active_params = false; // also see 2.6
    bool max_param_change_exceeds_tol; // boolean stopping criterion = stop iteration if params don't change much.
    double unregularized_optimal_param; // the value on `expression` inside the S(expression, \lambda\alpha) in equation (5)
    double tmp_new_param; // to hold the new params before we update the params vector
    double tmp_new_minus_old_param; // updated minus old params
    size_t curr_round; // we'll return the total number of rounds
    for(curr_round=0; curr_round<max_coord_descent_rounds; curr_round++){
        #if DEBUG
        std::cout<<"curr_round "<<curr_round
                 << " current_round_is_for_only_active_params "<< current_round_is_for_only_active_params<<std::endl;
        #endif
        max_param_change_exceeds_tol = false;
        // toggle all params to 'active' if we're starting a round where we update everything
        if(!current_round_is_for_only_active_params){
            #pragma omp parallel for schedule(static)
            for(size_t j=0; j<D; j++){
                inactive_params[j] = false;
            }
        }
        // do one round of coordinate descent, where we iterate through all params and update each in turn
        for(size_t j=0; j<D; j++){
            if(inactive_params[j]) continue;
            // compute the inside thing
            unregularized_optimal_param = params_unchecked[j];
            #pragma omp parallel for schedule(static) reduction(+:unregularized_optimal_param)
            for(size_t i=0; i<N; i++){
                unregularized_optimal_param += x_unchecked(j*N+i) * (resids[i] / N); // 
            }
            // if big enough, update the parameter, see equation(6) and (5)
            if(std::abs(unregularized_optimal_param) > l1_reg){
                if(unregularized_optimal_param > 0){
                    tmp_new_param = (unregularized_optimal_param - l1_reg) / (1 + l2_reg);
                }else{
                    tmp_new_param = (unregularized_optimal_param + l1_reg) / (1 + l2_reg);
                }
                //  update the resids because params changed
                tmp_new_minus_old_param = tmp_new_param - params_unchecked[j];
                // finally, update the params
                params_unchecked[j] = tmp_new_param;
            // if the unregularized new param is not big enough, set to zero (final case in equation (6))
            }else{
                tmp_new_minus_old_param = -params_unchecked[j];
                params_unchecked[j] = 0;
                // also, add it to the set of inactive params so we know not to check it next round
                inactive_params[j] = true;
            }
            // figure out if params changed enough
            max_param_change_exceeds_tol = (max_param_change_exceeds_tol) || (std::abs(tmp_new_minus_old_param) > tol);
            // now, update the residuals if we updated this param
            if(tmp_new_minus_old_param != 0){
                #pragma omp parallel for schedule(static)
                for(size_t i=0; i<N; i++){
                    resids[i] -= x_unchecked(j*N+i) * tmp_new_minus_old_param; // 
                }
            }
        }
        
        #if DEBUG // print params after each round
        for(size_t j=0; j<D; j++){
            std::cout<<params_unchecked[j]<<",";
        }
        std::cout<<std::endl;
        #endif
        //  if we're in an active set only round, then we should go back to an everything round
        //    if the tolerance change is satisfied.
        if(current_round_is_for_only_active_params){
            if(!max_param_change_exceeds_tol){
                current_round_is_for_only_active_params = false;
            }
        } else {
            // if we're in an update-everything round and nothing was updated enough, then we're finished
            if(!max_param_change_exceeds_tol){
                break;
            }
            // if we're not finished, then the next round we go back to doing only active set stuff
            current_round_is_for_only_active_params = true;
        }
    }

    // return the total number of rounds
    return curr_round;
}

PYBIND11_MODULE(enet_helpers, m){
    m.doc() = "elastic net";
    m.def("copy_input_x_data", &copy_input_x_data, 
        "copy x data for downstream use");
    m.def("compute_mean_std_and_standardize_x_data", &compute_mean_std_and_standardize_x_data, 
        "compute mean and std of each input x column, and then standardize the input x data");
    m.def("estimate_squaredloss_naive", &estimate_squaredloss_naive, 
        "coordinate descent optimization for elasticnet with squared loss, using the 'naive' update strategy");
}
