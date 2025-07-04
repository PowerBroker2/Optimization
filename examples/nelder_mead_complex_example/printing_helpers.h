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