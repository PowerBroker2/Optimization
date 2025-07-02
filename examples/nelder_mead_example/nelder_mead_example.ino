#include "Optimization.h"
#include "printing_helpers.h"


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