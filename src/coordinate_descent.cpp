#include "coordinate_descent.h"
#include "data.h"
#include "objective.h"
#include "common.h"
#include <cmath>
#include <memory>


// function to return smart ptr to an Objective function based on string input
std::unique_ptr<Objective> create_and_initialize_objective(const std::string& obj_str, 
                                                           const Data& data, 
                                                           const py::detail::unchecked_reference<double, 1> coefs_unchecked)
{
    // initialize the objective, will keep track of state information needed to update
    //  coefs that's specific to the particular objective itself
    std::unique_ptr<Objective> obj_ptr;
    if(obj_str=="l2")
    {
        obj_ptr = std::make_unique<L2Objective>(data, coefs_unchecked);
        #if DEBUG
        std::cout << "objective is l2" << std::endl;
        #endif
    }
    else
    {
        throw std::runtime_error("objective function not supported");
    }
    return obj_ptr;

}

// after we compute the optimal no-regularization coef, apply regularization
//  and return the regularized coef
double apply_l1_l2_reg_to_unregularized_coef(double unregularized_optimal_coef_j,
                                             double l1_reg, double l2_reg)
{
    // if big enough, update the coef, see equation(6) and (5)
    if(std::abs(unregularized_optimal_coef_j) > l1_reg)
    {
        if(unregularized_optimal_coef_j > 0){
            return (unregularized_optimal_coef_j - l1_reg) / (1 + l2_reg);
        } 
        else 
        {
            return (unregularized_optimal_coef_j + l1_reg) / (1 + l2_reg);
        }
    } 
    else
    { // if not, then coef gets set to 0
        return 0;
    }
}

int cyclic_coordinate_descent(Data& data, std::string& obj_str,
                              py::array_t<double> intercept, py::array_t<double> coefs,
                              py::array_t<double> coefs_init, 
                              double lambda, double alpha, 
                              double tol, size_t max_coord_descent_rounds,
                              int num_threads)
{
    omp_set_num_threads(num_threads);
    size_t D = data.D;

    // compute the total l1 and l2 reg
    double l1_reg = alpha*lambda;
    double l2_reg = (1-alpha)*lambda;

    // initialize the final coefs at the value of the initial coefs
    auto coefs_init_unchecked = coefs_init.unchecked<1>();
    auto coefs_unchecked = coefs.mutable_unchecked<1>();
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<D; j++)
    {
        coefs_unchecked[j] = coefs_init_unchecked[j];
    }

    // initialize the objective
    std::unique_ptr<Objective> obj_ptr = create_and_initialize_objective(obj_str, data, coefs_unchecked);

    // store the intercept computed during objective initialization
    auto intercept_unchecked = intercept.mutable_unchecked<1>();
    intercept_unchecked[0] = obj_ptr->_intercept;

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
    double unregularized_optimal_coef_j; // the value on `expression` inside the S(expression, \lambda\alpha) in equation (5)
    double new_coef_j; // to hold the new coefs before we update the coefs vector
    size_t curr_round; // we'll also want to keep track of / return the total number of rounds
    for(curr_round=0; curr_round<max_coord_descent_rounds; curr_round++)
    {
        #if DEBUG
        std::cout<<"curr_round " << curr_round
                 << " curr_round_ignore_inactive " 
                 << curr_round_ignore_inactive
                 << std::endl;
        #endif
        max_coef_change_exceeds_tol = false;
        // toggle all coefs to 'active' if we're starting a round where we update everything
        if(!curr_round_ignore_inactive)
        {
            #pragma omp parallel for schedule(static)
            for(size_t j=0; j<D; j++)
            {
                coef_is_inactive[j] = false;
            }
        }
        // do one round of coordinate descent, where we iterate through all coefs and update each in turn
        for(size_t j=0; j<D; j++){
            if(coef_is_inactive[j])
            {
                continue;
            }
            // compute the unregularized_optimal_coef_j
            unregularized_optimal_coef_j = obj_ptr->get_unregularized_optimal_coef_j(j);
            // apply regularization adjustment to this unregularized_optimal_coef_j
            new_coef_j = apply_l1_l2_reg_to_unregularized_coef(unregularized_optimal_coef_j, l1_reg, l2_reg);
            // update the objective's internal state if we updated this coef
            if(new_coef_j != coefs_unchecked[j])
            {
                obj_ptr->update_internal_state_after_coef_update(j, new_coef_j);
            }
            // deactive the coef if it is now zero
            if(new_coef_j == 0)
            {
                coef_is_inactive[j] = true;
            }
            // keep track of the max that any coef this round has changed
            max_coef_change_exceeds_tol = (max_coef_change_exceeds_tol)
                                            || (std::abs(new_coef_j - coefs_unchecked[j]) > tol);
            // finally, update the coef
            coefs_unchecked[j] = new_coef_j;
        }
        
        #if DEBUG // print coefs after each round
        for(size_t j=0; j<D; j++)
        {
            std::cout<<coefs_unchecked[j]<<",";
        }
        std::cout<<std::endl;
        #endif
        //  if we're in an active set only round, and coefs didn't change much,
        //    then we've exhausted updates to this active set and should go back to updating everything
        if(curr_round_ignore_inactive)
        {
            if(!max_coef_change_exceeds_tol)
            {
                curr_round_ignore_inactive = false;
            }
        }
        else
        {
            // if we're in an update-everything round and nothing was updated enough, then we're finished
            if(!max_coef_change_exceeds_tol)
            {
                break;
            }
            // if coefs changed a bit, then we continue with the cordinate descent, in that we go back
            //   to only updating the active coefs
            curr_round_ignore_inactive = true;
        }
    }
    // return the total number of rounds
    return curr_round;
}
