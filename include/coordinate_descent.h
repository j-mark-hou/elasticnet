#ifndef COORDINATE_DESCENT_H_
#define COORDINATE_DESCENT_H_

#include "common.h"
#include "data.h"
#include <vector>


// coordinate descent optimization for elasticnet with squared loss, using the 'naive' update strategy
//   (as opposed to the 'covariance' update strategy) 
// tol = when to terminate convergence if parameters don't change too much
// coefs_init = where to start estimation at
// coefs = array for holding the final estimated coefs.  no intercept returned because it's aways just the avg y 
// max_coord_descent_rounds = how many rounds of coordinate descent (1 round = going through all coords once) to do at max
// lambda = the total regularization amount
// alpha = the fraction of regularization that goes on the L1 term (so 1-alpha goets on l2)
int estimate_squaredloss_naive(Data& data,
                               py::array_t<double> params_init, py::array_t<double> params,
                               double lambda, double alpha, 
                               double tol, size_t max_coord_descent_rounds,
                               int num_threads);

#endif // COORDINATE_DESCENT_H_