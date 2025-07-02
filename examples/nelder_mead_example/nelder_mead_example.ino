#include "Optimization.h"


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


double function(const VectorXd& x)
{
    VectorXd x_real(3);
    x_real << 0, 1, 2;

    VectorXd diff(3);
    diff = x_real - x;

    return diff.norm();
}


void setup()
{
    Serial.begin(115200);

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);
    
    VectorXd x_start(3);
    x_start << 2, 2, 2;
    
    auto res = Nelder_Mead_Optimizer(function, x_start, 0.1, 10e-10);
    
    Serial.println();
    Serial.println("Starting Vector");
    printVecXd(x_start);
    Serial.println();
    Serial.println("Optimized Vector");
    printVecXd(res);
    Serial.println();
    Serial.println("Optimized Vector Score (closer to 0 is better)");
    Serial.println(function(res));
}


void loop()
{
    
}