#include "Optimization.h"


double function(const VectorXd& x)
{
    VectorXd x_real(3);
    x_real << 0, 1, 2;

    VectorXd cross_product(3);
    cross_product << ( (x(1) * x_real(2)) - (x(2) * x_real(1))),
                     (-(x(0) * x_real(2)) + (x(2) * x_real(0))),
                     ( (x(0) * x_real(1)) - (x(1) * x_real(0)));
    
    Serial.println();
    printVecXd(cross_product);
    Serial.println();

    return cross_product.norm();
}

void setup()
{
    Serial.begin(115200);

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);
    
    VectorXd x_start(3);
    x_start << 0, 1, 2;
    
    MatrixXd mat(3, 3);
    mat << 1, 0, 0,
           0, 1, 0,
           0, 0, 1;
    
    Serial.println("x_start");
    printVecXd(x_start);
    Serial.println();
    Serial.println("init simplex args");
    printMatXd(init_simplex_args(x_start));
    Serial.println();
    Serial.println("init simplex results");
    printVecXd(get_simplex_results(function,
                                   init_simplex_args(x_start)));

    //auto res = Nelder_Mead_Optimizer(function, x_start, 0.1, 10e-10);
}

void loop()
{
    
}