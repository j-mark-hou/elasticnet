#ifndef OBJECTIVE_H_
#define OBJECTIVE_H_

#include "common.h"
#include "data.h"
#include <pybind11/numpy.h>
#include <string>

// abstract base class that all objectives should subclass
class Objective
{
public:
    // something to hold the computed intercept
    double _intercept;
    // When we initialize the objective, we'll also attach to it some references
    //  to the data and coefs, so that we can easily update the state wthout having
    //  to re-pass-in these each timeto the methods below.
    // This method should also compute the intercept and store it in _intercept.
    Objective(const Data& data, const py::detail::unchecked_reference<double, 1> coefs_unchecked) 
        : _data(data), _coefs_unchecked(coefs_unchecked){}
    virtual ~Objective(){}
    // the intercept is not regularized, implement a function to return it
    virtual double get_unregularized_optimal_intercept() = 0;
    // the l2 and l1 regularization adjustment is shared across all objectives, whereas
    //  the unregularized optimal coef is unique to a particular objective function,
    //  which is why this is here
    virtual double get_unregularized_optimal_coef_j(size_t j) = 0;
    // after we update the coefs, we may want to update the internal state
    //   that the objective is tracking (e.g. the second-order approx of the
    //   objective function at the updated parameter values)
    virtual void update_internal_state_after_coef_update(size_t j, double new_coef_j) = 0;
protected:
    const Data& _data;
    const py::detail::unchecked_reference<double, 1> _coefs_unchecked;
};


// squared loss objective.  as in, linear regression.
class L2Objective : public Objective
{
public:
    L2Objective(const Data& data, const py::detail::unchecked_reference<double, 1> coefs_unchecked)
    : Objective(data, coefs_unchecked)
    {
        // compute intercept
        _intercept = get_unregularized_optimal_intercept();
        // initialize residuals at y - intercept
        _resids = std::vector<double>(_data.N);
        for(size_t i=0; i<_data.N; i++)
        {
            _resids[i] = _data.y[i] - _intercept;
        }
        // and then subtract the x_ij * params[j] repeatedly.
        // do this column by column, but parallelize within columns
        for(size_t j=0; j<_data.D; j++)
        {
            #pragma omp parallel for schedule(static)
            for(size_t i=0; i<_data.N; i++)
            {
                _resids[i] -= _data.x[j*_data.N+i] * _coefs_unchecked[j]; //
            } 
        }
    }

    double get_unregularized_optimal_intercept()
    {
        // in the linear regression case, the coef is just the mean of the y
        double intercept=0;
        int n = 1;
        for(size_t i=0; i<_data.N; i++)
        {
            intercept += (_data.y[i]-intercept)/n;
            n++;
        }
        return intercept;
    }

    double get_unregularized_optimal_coef_j(size_t j)
    {
        double unregularized_optimal_coef_j = _coefs_unchecked[j];
        #pragma omp parallel for schedule(static) reduction(+:unregularized_optimal_coef_j)
        for(size_t i=0; i<_data.N; i++)
        {
            unregularized_optimal_coef_j += _data.x[j*_data.N+i] * (_resids[i] / _data.N);
        }
        return unregularized_optimal_coef_j;
    }

    void update_internal_state_after_coef_update(size_t j, double new_coef_j)
    {
        double new_minus_old_coef_j = new_coef_j - _coefs_unchecked[j];
        #pragma omp parallel for schedule(static)
        for(size_t i=0; i<_data.N; i++)
        {
            _resids[i] -= _data.x[j*_data.N+i] * new_minus_old_coef_j; // 
        }
    }
    
private:
    // compute the initial residuals (r in equation (7)), which is (y - yhat)
    // note that this number is never re-computed from first principles, rather
    // it's updated each time a param changes by subtracting the old and adding the new
    // TODO: do we need to worry about the value drifting further and further from truth?
    std::vector<double> _resids;
};

#endif //OBJECTIVE_H_
