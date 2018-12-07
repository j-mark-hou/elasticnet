#ifndef OBJECTIVE_H_
#define OBJECTIVE_H_

#include "common.h"
#include "data.h"
#include <pybind11/numpy.h>


class L2Objective {
public:
    L2Objective(Data& data, py::detail::unchecked_reference<double, 1> coefs_unchecked) :
                                    _data(data), _coefs_unchecked(coefs_unchecked){
        _resids = std::vector<double>(_data.N);
        // initialize residuals at y
        for(size_t i=0; i<_data.N; i++){
            _resids[i] = _data.y[i];
        }
        // and then subtract the x_ij * params[j] repeatedly.
        // do this column by column, but parallelize within columns
        for(size_t j=0; j<_data.D; j++){
            #pragma omp parallel for schedule(static)
            for(size_t i=0; i<_data.N; i++){
                _resids[i] -= _data.x[j*_data.N+i] * _coefs_unchecked[j]; //
            } 
        }
    };
    // the l2 and l1 regularization adjustment is shared across all objectives, whereas
    //  the unregularized optimal coef is unique to a particular objective function,
    //  which is why this is here
    double get_unregularized_optimal_coef(size_t j){
        double unregularized_optimal_coef = _coefs_unchecked[j];
        #pragma omp parallel for schedule(static) reduction(+:unregularized_optimal_coef)
        for(size_t i=0; i<_data.N; i++){
            unregularized_optimal_coef += _data.x[j*_data.N+i] * (_resids[i] / _data.N); // 
        }
        return unregularized_optimal_coef;
    }
    // after we update the coefs, we may want to update the internal state
    //   that the objective is tracking.
    void update_internal_state_after_coef_update(size_t j, double new_coef_j){
        double new_minus_old_coef_j = new_coef_j - _coefs_unchecked[j];
        #pragma omp parallel for schedule(static)
        for(size_t i=0; i<_data.N; i++){
            _resids[i] -= _data.x[j*_data.N+i] * new_minus_old_coef_j; // 
        }
    }
private:
    // compute the initial residuals (r in equation (7)), which is (y - yhat)
    // note that this number is never re-computed from first principles, rather
    // it's updated each time a param changes by subtracting the old and adding the new
    // TODO: do we need to worry about the value drifting further and further from truth?
    std::vector<double> _resids;
    Data& _data;
    py::detail::unchecked_reference<double, 1> _coefs_unchecked;
};


#endif //OBJECTIVE_H_