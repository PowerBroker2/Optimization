#include "Optimization.h"
#include "printing_helpers.h"


constexpr int num_samples = 1000;


MatrixXd true_sensor_data(3, num_samples);
MatrixXd noised_sensor_data(3, num_samples);
Matrix3d noise_mat;


// Function to generate points on a sphere using the Fibonacci method
// Returns MxN matrix of points on sphere where M=3 and N=samples
MatrixXd fibonacci_sphere(const int&    samples = 1000,
                          const double& radius  = 1.0)
{
    MatrixXd points(3, samples);
    const double phi = M_PI * (sqrt(5.0) - 1.0);  // golden angle in radians

    for (int i = 0; i < samples; ++i)
    {
        double y           = (1.0 - (i / static_cast<double>(samples - 1)) * 2.0);  // y goes from 1 to -1
        double temp_radius = radius * sqrt(1.0 - y * y);  // radius at y

        double theta = phi * i;

        double x = cos(theta) * temp_radius;
        double z = sin(theta) * temp_radius;

        points(all, i) << x, y, z;
    }

    return points;
}


// https://stackoverflow.com/a/15142446
MatrixXd cov(MatrixXd mat)
{
    MatrixXd centered = mat.rowwise() - mat.colwise().mean();
    return (centered.adjoint() * centered) / double(mat.rows() - 1);
}


double function(const VectorXd& vec)
{
    Map<const MatrixXd> cal_mat(vec.data(), 3, 3);
    Serial.println();
    Serial.println("cal_mat");
    printMatXd(cal_mat);

    Serial.println();
    Serial.println("noised_sensor_data");
    printMatXd(noised_sensor_data);
    Serial.println();
    Serial.println("cal_data_test");
    printMatXd(cal_mat * noised_sensor_data);

    MatrixXd cal_data;
    cal_data = cal_mat * noised_sensor_data;
    Serial.println();
    Serial.println("cal_data_real");
    printMatXd(cal_data);

    Matrix3d covariance;
    covariance = cov(cal_data); // TODO: This is screwed
    Serial.println();
    Serial.println("covariance");
    printMatXd(covariance);

    Matrix3d eye3;
    eye3 << 1, 0, 0,
            0, 1, 0,
            0, 0, 1;
    
    Matrix3d diff;
    diff << covariance - eye3;
    Serial.println();
    Serial.println("diff");
    printMatXd(diff);

    Serial.println();

    return diff.norm();
}


void setup()
{
    Serial.begin(115200);

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    noise_mat << 2,  1, -1,
                 5, -1,  3,
                -2,  2,  1;

    true_sensor_data   << fibonacci_sphere(num_samples, 1);
    noised_sensor_data << noise_mat * true_sensor_data;

    Serial.println("true_sensor_data");
    printMatXd(true_sensor_data);
    Serial.println();
    Serial.println("noised_sensor_data");
    printMatXd(noised_sensor_data);
    
    Matrix3d start_cal_mat;
    start_cal_mat << 1, 0, 0,
                     0, 1, 0,
                     0, 0, 1;
    Map<const VectorXd> start_cal_vec(start_cal_mat.data(),
                                      start_cal_mat.size());
    
    // VectorXd res = Nelder_Mead_Optimizer(function,
    //                                      start_cal_vec,
    //                                      0.1,
    //                                      10e-10);
    
    // Map<const MatrixXd> res_cal_mat(res.data(), 3, 3);

    // Serial.println();
    // Serial.println("Starting Cal Mat");
    // printMatXd(start_cal_mat);
    // Serial.println();
    // Serial.println("Optimized Cal Mat");
    // printMatXd(res_cal_mat);
    // Serial.println();
    // Serial.println("Optimized Cal Score (closer to 0 is better)");
    // Serial.println(function(res));
}


void loop()
{
    
}


// def plot(x, y, z):
//     fig = plt.figure(figsize=(10, 7))
//     ax = fig.add_subplot(111, projection='3d')
//     ax.scatter(x, y, z, c='red', marker='o')
//     ax.set_xlabel("X")
//     ax.set_ylabel("Y")
//     ax.set_zlabel("Z")
//     ax.set_box_aspect([1, 1, 1])
//     ax.view_init(elev=20, azim=40)
//     plt.title("3D Scatter Plot of Dataset 2")
//     plt.grid(True)
//     plt.show(block=False)