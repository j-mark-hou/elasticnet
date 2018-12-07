#include "coordinate_descent.h"
#include "data.h"
#include "objective.h"
#include "common.h"

#include <cmath>
#include <string>


// after we compute the optimal no-regularization coef, apply regularization
//  and return the regularized coef
double apply_l1_l2_reg_to_unregularized_coef(double unregularized_optimal_coef,
                                             double l1_reg, double l2_reg){
    // if big enough, update the coef, see equation(6) and (5)
    if(std::abs(unregularized_optimal_coef) > l1_reg){
        if(unregularized_optimal_coef > 0){
            return (unregularized_optimal_coef - l1_reg) / (1 + l2_reg);
        }else{
            return (unregularized_optimal_coef + l1_reg) / (1 + l2_reg);
        }
    } else { // if not, then coef gets set to 0
        return 0;
    }
}

int estimate_squaredloss_naive(Data& data,
                               py::array_t<double> coefs_init, py::array_t<double> coefs,
                               double lambda, double alpha, 
                               double tol, size_t max_coord_descent_rounds,
                               int num_threads){
    omp_set_num_threads(num_threads);
    size_t N = data.N, D = data.D;

    // compute the total l1 and l2 reg
    double l1_reg = alpha*lambda;
    double l2_reg = (1-alpha)*lambda;

    // initialize the final coefs at the value of the initial coefs
    auto coefs_init_unchecked = coefs_init.unchecked<1>();
    auto coefs_unchecked = coefs.mutable_unchecked<1>();
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<D; j++){
        coefs_unchecked(j) = coefs_init_unchecked(j);
    }

    // initialize the objective, will keep track of state information needed to update
    //  coefs that's specific to the particular objective itself
    L2Objective obj(data, coefs_unchecked);

    // compute the initial residuals (r in equation (7)), which is (y - yhat)
    // note that this number is never re-computed from first principles, rather
    // it's updated each time a coef changes by subtracting the old and adding the new
    // TODO: do we need to worry about the value drifting further and further from truth?
    // std::vector<double> resids(N);

    // // initialize residuals at y
    // for(size_t i=0; i<N; i++){
    //     resids[i] = data.y[i];
    // }
    // // and then subtract the x_ij * coefs[j] repeatedly.
    // for(size_t j=0; j<D; j++){
    //     #pragma omp parallel for schedule(static) // can't parallelize above as all i's get updated for each j
    //     for(size_t i=0; i<N; i++){
    //         resids[i] -= data.x[j*N+i] * coefs_unchecked[j]; //
    //     } 
    // }


    // estimate by active-set iteration (see section 2.6), which amounts to these two steps:
    //  1. loop through all D coefs once
    //  2. then repeately loop through just the active coefs until no change
    //    - coords are removed from the active set as they become zero
    //  3. repeat
    //  4. terminate when a loop through all coefs fails
    // some state variables to keep track of stuff
    std::vector<bool> coef_is_inactive(D);
    bool curr_round_ignore_inactive = false; // also see 2.6
    bool max_coef_change_exceeds_tol; // boolean stopping criterion = stop iteration if coefs don't change much.
    double unregularized_optimal_coef; // the value on `expression` inside the S(expression, \lambda\alpha) in equation (5)
    double new_coef_j; // to hold the new coefs before we update the coefs vector
    size_t curr_round; // we'll also want to keep track of / return the total number of rounds
    for(curr_round=0; curr_round<max_coord_descent_rounds; curr_round++){
        #if DEBUG
        std::cout<<"curr_round " << curr_round
                 << " curr_round_ignore_inactive " 
                 << curr_round_ignore_inactive
                 << std::endl;
        #endif
        max_coef_change_exceeds_tol = false;
        // toggle all coefs to 'active' if we're starting a round where we update everything
        if(!curr_round_ignore_inactive){
            #pragma omp parallel for schedule(static)
            for(size_t j=0; j<D; j++){
                coef_is_inactive[j] = false;
            }
        }
        // do one round of coordinate descent, where we iterate through all coefs and update each in turn
        for(size_t j=0; j<D; j++){
            if(coef_is_inactive[j]){
                continue;
            }
            unregularized_optimal_coef = obj.get_unregularized_optimal_coef(j);

            // // compute the unregularized_optimal_coef
            // unregularized_optimal_coef = coefs_unchecked[j];
            // #pragma omp parallel for schedule(static) reduction(+:unregularized_optimal_coef)
            // for(size_t i=0; i<N; i++){
            //     unregularized_optimal_coef += data.x[j*N+i] * (resids[i] / N); // 
            // }

            // apply regularization adjustment to this unregularized_optimal_coef 
            //  and then update some iteration state variables

            new_coef_j = apply_l1_l2_reg_to_unregularized_coef(unregularized_optimal_coef, l1_reg, l2_reg);
            coefs_unchecked[j] = new_coef_j;
            if(coefs_unchecked[j] == 0){
                coef_is_inactive[j] = true;
            }
            max_coef_change_exceeds_tol = (max_coef_change_exceeds_tol)
                                            || (std::abs(new_coef_j - coefs_unchecked[j]) > tol);

            // // update the residuals if we updated this param
            // if(new_coef_j != coefs_unchecked[j]){
            //     #pragma omp parallel for schedule(static)
            //     for(size_t i=0; i<N; i++){
            //         resids[i] -= data.x[j*N+i] * tmp_new_minus_old_param; // 
            //     }
            // }
            obj.update_internal_state_after_coef_update(j, new_coef_j);
        }
        
        #if DEBUG // print coefs after each round
        for(size_t j=0; j<D; j++){
            std::cout<<coefs_unchecked[j]<<",";
        }
        std::cout<<std::endl;
        #endif
        //  if we're in an active set only round, and coefs didn't change much,
        //    then we've exhausted updates to this active set and should go back to updating everything
        if(curr_round_ignore_inactive){
            if(!max_coef_change_exceeds_tol){
                curr_round_ignore_inactive = false;
            }
        } else {
            // if we're in an update-everything round and nothing was updated enough, then we're finished
            if(!max_coef_change_exceeds_tol){
                break;
            }
            // if coefs changed a bit, then we continue with the cordinate descent, so that the
            //   to doing only active set stuff
            curr_round_ignore_inactive = true;
        }
    }

    // return the total number of rounds
    return curr_round;
}
