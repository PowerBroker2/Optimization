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




MatrixXd init_simplex_args(const VectorXd& x_start,
                           const double&   step = 0.1)
{
    int dim = x_start.size();

    MatrixXd simplex_args(dim, dim + 1);
    simplex_args.col(0) = x_start;

    for (int i=1; i<dim; i++) // ??????
    {
        VectorXd step_vec(dim);
        step_vec = VectorXd::Zero(dim);
        step_vec(i-1) = step;

        simplex_args.col(i) = x_start + step_vec;
    }

    return simplex_args;
}




VectorXd get_simplex_results(      double    (*func)(const VectorXd&),
                             const MatrixXd& simplex_args)
{
    int dim = simplex_args.cols();

    VectorXd simplex_results(dim);

    for (int i=0; i<dim; i++)
        simplex_results(i) = func(simplex_args.col(i));

    return simplex_results;
}




void sort_args_and_results(MatrixXd& simplex_args,
                           VectorXd& simplex_results)
{
    VectorXi sort_order = find_vec_ord(simplex_results);

    // Sort highest cost in leftmost column and lowest cost in rightmost column
    simplex_args    = sort_cols(simplex_args, sort_order);
    simplex_results = sort_cols((MatrixXd)simplex_results, sort_order); // ??????
}




VectorXd calc_centroid(const MatrixXd& sorted_simplex_args)
{
    // Average of all args except for one with largest result
    // Simplex arguments MUST be sorted beforehand
    return sorted_simplex_args(all, seq(1, last)).rowwise().mean();
}




VectorXd get_best_args(const MatrixXd& simplex_args)
{
    // Simplex arguments MUST be sorted beforehand
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
