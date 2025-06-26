// Credit:
// https://github.com/Enderdead/nelder-mead/blob/master/NelderMeadOptimizer.h

#pragma once
#include "Arduino.h"
#include "eigen.h"
#include <Eigen/Dense>




using namespace Eigen;


void printVecXd(const VectorXd& vec,
                const int&      p=5,
                Stream&         stream=Serial)
{
    for (int i=0; i<vec.rows(); i++)
    {
        if (vec(i) >= 0)
            Serial.print(' ');
        
        stream.println(vec(i), p);
    }
}


void printVecXi(const VectorXi& vec,
                Stream&         stream=Serial)
{
    for (int i=0; i<vec.rows(); i++)
    {
        if (vec(i) >= 0)
            Serial.print(' ');
        
        stream.println(vec(i));
    }
}


void printMatXd(const MatrixXd& mat,
                const int&      p=5,
                Stream&         stream=Serial)
{
    for (int i=0; i<mat.rows(); i++)
    {
        for (int j=0; j<mat.cols(); j++)
        {
            if (mat(i, j) >= 0)
                Serial.print(' ');
        
            stream.print(mat(i, j), p);

            if (j != (mat.cols() - 1))
                stream.print(", ");
        }

        stream.println();
    }
}




/**
 * @brief Find order of each element based on value.
 *
 * This function takes a vector "vec" and ranks each
 * value at each element. The ranking/ordering is such
 * that the highest value of "vec" will be given the
 * ranking/ordering of 0 and the lowest value of "vec"
 * will be given the value of n-1 where n is the
 * number of elements in "vec".
 *
 * @param vec The vector of whos elements need to be ranked/ordered.
 * @return    A vector of integers denoting the ranking/ordering of each value in "vec".
 */
VectorXi find_vec_ord(const VectorXd& vec)
{
    int dim = vec.size();

    VectorXi order(dim);

    for (int i=0; i<dim; i++)
    {
        int numBigger = 0;

        for (int k=0; k<dim; k++)
        {
            if (vec(i) < vec(k))
                numBigger++;
        }

        order(i) = numBigger;
    }

    return order;
}




/**
 * @brief Reorder columns based on given column order.
 *
 * This function takes a matrix "mat" and vector of
 * column orders "order" and sorts the columns of "mat"
 * such that the ith column of "mat" should be put in
 * the order(ith) location in the returned, reordered
 * matrix "ordered_mat".
 *
 * @param mat   The matrix whos columns are to be reordered.
 * @param order The vector detailing which location the corresponding column should be placed in.
 * @return      The reordered matrix.
 */
MatrixXd sort_cols(const MatrixXd& mat, const VectorXi& order)
{
    int rows = mat.rows();
    int cols = mat.cols();

    MatrixXd ordered_mat(rows, cols);

    int dim = order.size();

    for (int i=0; i<dim; i++)
        ordered_mat.col(order(i)) = mat.col(i);

    return ordered_mat;
}




/**
 * @brief Reorder elements based on given order.
 *
 * This function takes a vector "vec" and vector of
 * column orders "order" and sorts the elements of "vec"
 * such that the ith value of "vec" should be put in
 * the order(ith) location in the returned, reordered
 * vector "ordered_vec".
 *
 * @param vec   The vector whos elements are to be reordered.
 * @param order The vector detailing which location the corresponding element should be placed in.
 * @return      The reordered vector.
 */
VectorXd sort_elements(const VectorXd& vec, const VectorXi& order)
{
    int dim = vec.size();

    VectorXd ordered_vec(dim);

    for (int i=0; i<dim; i++)
        ordered_vec(order(i)) = vec(i);

    return ordered_vec;
}




/**
 * @brief Initializes a matrix of dithered, "simplex" arguments.
 *
 * This function takes an initial argument guess, dithers the arguments
 * by a given step size, and returns the group of dithered, "simplex"
 * arguments as a matrix. This matrix is then used to bootstrap the
 * optimization from an initial guess.
 *
 * @param x_start A vector of "initial guess" arguments.
 * @param step    The argument dither step size.
 * @return        The dithered, "simplex" arguments.
 */
MatrixXd init_simplex_args(const VectorXd& x_start,
                           const double&   step = 0.1)
{
    int dim = x_start.size();

    MatrixXd simplex_args(dim, dim + 1);
    simplex_args.col(0) = x_start;

    for (int i=1; i<(dim+1); i++)
    {
        VectorXd step_vec(dim);
        step_vec = VectorXd::Zero(dim);
        step_vec(i-1) = step;

        simplex_args.col(i) = x_start + step_vec;
    }

    return simplex_args;
}




/**
 * @brief Evaluates the given system function for each "simplex" argument.
 *
 * This function evaluates the given system function for each "simplex" argument and
 * returns the result as a matrix.
 *
 * @param func         Given system function for the arguments to be evaluated at.
 * @param simplex_args The "simplex" arguments.
 * @return             The results of evaluating the given system function at the "simplex" arguments.
 */
VectorXd get_simplex_results(      double    (*func)(const VectorXd&),
                             const MatrixXd& simplex_args)
{
    int dim = simplex_args.cols();

    VectorXd simplex_results(dim);

    for (int i=0; i<dim; i++)
        simplex_results(i) = func(simplex_args.col(i));

    return simplex_results;
}




/**
 * @brief Sort the simplex arguments and function outputs by function output.
 *
 * This function sorts the simplex arguments and function outputs such
 * that the argument that produces the smallest function output is last
 * (rightmost col) and the argument that produces the largest function
 * output is first (leftmost col). Later, the simplex argument with the
 * largest function output will then be updated to move the entire simplex
 * to convergence at the function's minimum value.
 *
 * @param simplex_args    The "simplex" arguments.
 * @param simplex_results The sorted "simplex" arguments.
 * @return                None.
 */
void sort_args_and_results(MatrixXd& simplex_args,
                           VectorXd& simplex_results)
{
    VectorXi sort_order = find_vec_ord(simplex_results);

    // Sort highest cost in leftmost column and lowest cost in rightmost column
    simplex_args    = sort_cols(simplex_args, sort_order);
    simplex_results = sort_elements(simplex_results, sort_order);
}




/**
 * @brief Find centroid of all simplex args except for the one with largest result.
 *
 * This function finds centroid of all simplex args except for the one with
 * largest result. Simplex arguments MUST be sorted beforehand.
 *
 * @param sorted_simplex_args The sorted "simplex" arguments.
 * @return                    The vector representing the simplex's centroid coordinate.
 */
VectorXd calc_centroid(const MatrixXd& sorted_simplex_args)
{
    return sorted_simplex_args(all, seq(1, last)).rowwise().mean();
}




/**
 * @brief Find the best simplex argument in the sorted argument list.
 *
 * This function finds the best simplex argument in the sorted
 * argument list. Simplex arguments MUST be sorted beforehand.
 *
 * @param sorted_simplex_args The sorted "simplex" arguments.
 * @return                    The vector of the best simplex argument.
 */
VectorXd get_best_args(const MatrixXd& simplex_args)
{
    return simplex_args(all, last);
}




double get_best_result(const VectorXd& simplex_results)
{
    // Simplex results MUST be sorted beforehand
    return simplex_results(last);
}




VectorXd get_next_best_args(const MatrixXd& simplex_args)
{
    // Simplex arguments MUST be sorted beforehand
    return simplex_args(all, last-1);
}




double get_next_best_result(const VectorXd& simplex_results)
{
    // Simplex results MUST be sorted beforehand
    return simplex_results(last-1);
}




VectorXd get_worst_args(const MatrixXd& simplex_args)
{
    // Simplex arguments MUST be sorted beforehand
    return simplex_args(all, 0);
}




double get_worst_result(const VectorXd& simplex_results)
{
    // Simplex results MUST be sorted beforehand
    return simplex_results(0);
}




void update_worst_case(VectorXd& new_args,
                       double&   new_score,
                       MatrixXd& simplex_args,
                       VectorXd& simplex_results)
{
    simplex_args(all, last) = new_args;
    simplex_results(simplex_results.size() - 1) = new_score; // ??????
}




void shrink_args(const double&   sigma,
                       MatrixXd& simplex_args,
                       VectorXd& simplex_results,
                       double    (*func)(const VectorXd&))
{
    // Simplex arguments MUST be sorted beforehand
    int dim = simplex_args.cols();

    VectorXd best_args = get_best_args(simplex_args);

    for (int i=0; i<dim-1; i++) // Don't mess with the best args
        simplex_args(all, i) = (sigma * (simplex_args(all, i) - best_args)) + best_args;
    
    simplex_results = get_simplex_results(func, simplex_args);
}




VectorXd Nelder_Mead_Optimizer(      double   (*func)(const VectorXd&), // Function to minimize
                               const VectorXd& x_start,                 // Initial position
                               const double&   step            = 0.1,   // Look-around radius in initial step
                               const double&   no_improve_thr  = 10e-6, // Threshold on improve classification
                               const int&      no_improv_break = 10,    // Break after no_improv_break iterations without improvement
                               const int&      max_iter        = 0,     // Break after exeed max_iter iterations
                               const double&   alpha           = 1.0,   // Reflection multiplier
                               const double&   gamma           = 2.0,   // Expansion multiplier
                               const double&   rho             = -0.5,  // Contraction multiplier
                               const double&   sigma           = 0.5)   // Shrink multiplier
{
    MatrixXd simplex_args    = init_simplex_args(x_start, step);
    VectorXd simplex_results = get_simplex_results(func, simplex_args);
    sort_args_and_results(simplex_args, simplex_results);

    VectorXd centroid = calc_centroid(simplex_args);

    VectorXd best_args        = get_best_args(simplex_args);
    double   cur_best_result  = get_best_result(simplex_results);
    double   prev_best_result = cur_best_result;
    VectorXd next_best_args   = get_next_best_args(simplex_args);
    double   next_best_result = get_next_best_result(simplex_results);
    VectorXd worst_args       = get_worst_args(simplex_args);
    // double   worst_result     = get_worst_result(simplex_results);

    int iteration = 0;
    int no_improv = 0;

    while(true)
    {
        // Getting new results is already handled in `update_worst_case()` and `shrink_args()`
        sort_args_and_results(simplex_args, simplex_results);

        best_args        = get_best_args(simplex_args);
        prev_best_result = cur_best_result;
        cur_best_result  = get_best_result(simplex_results);
        next_best_args   = get_next_best_args(simplex_args);
        next_best_result = get_next_best_result(simplex_results);
        worst_args       = get_worst_args(simplex_args);
        // worst_result     = get_worst_result(simplex_results);

        if (max_iter && (iteration >= max_iter))
            return best_args;
        
        iteration++;

        if (cur_best_result < (prev_best_result - no_improve_thr))
            no_improv = 0;
        else
            no_improv++;
        
        if (no_improv >= no_improv_break)
            return best_args;

        centroid = calc_centroid(simplex_args);

        VectorXd reflection_pt    = (alpha * (centroid - worst_args)) + centroid;
        double   reflection_score = func(reflection_pt);

        if ((reflection_score < next_best_result) && (reflection_score >= cur_best_result))
        {
            // save reflection_pt
            update_worst_case(reflection_pt,
                              reflection_score,
                              simplex_args,
                              simplex_results);
            continue;
        }

        if (reflection_score < cur_best_result)
        {
            VectorXd expansion_pt    = (gamma * (centroid - worst_args)) + centroid;
            double   expansion_score = func(expansion_pt);

            if (expansion_score < reflection_score)
            {
                // save expansion_pt
                update_worst_case(expansion_pt,
                                  expansion_score,
                                  simplex_args,
                                  simplex_results);
                continue;
            }
            else
            {
                // save reflection_pt
                update_worst_case(reflection_pt,
                                  reflection_score,
                                  simplex_args,
                                  simplex_results);
                continue;
            }
        }

        VectorXd contraction_pt    = (rho * (centroid - worst_args)) + centroid;
        double   contraction_score = func(contraction_pt);

        if (contraction_score < next_best_result)
        {
            // save contraction_pt
            update_worst_case(contraction_pt,
                              contraction_score,
                              simplex_args,
                              simplex_results);
            continue;
        }

        // Shrink points
        shrink_args(sigma,
                    simplex_args,
                    simplex_results,
                    func);
    }

    return simplex_args(all, last);
}
