#include "Optimization.h"


double function(const VectorXd& x)
{
  return 0;
}

void setup()
{
  Serial.begin(115200);
  
  VectorXd x_start(4);
  x_start << 2, 1, 3, 4;
  
  MatrixXd mat(3, 4);
  mat << 2, 1, 3, 4,
         5, 6, 7, 8,
         4, 3, 2, 1;
  
  printMatXd(init_simplex_args(x_start));

  auto res = Nelder_Mead_Optimizer(function, x_start, 0.1, 10e-10);
}

void loop()
{
  
}