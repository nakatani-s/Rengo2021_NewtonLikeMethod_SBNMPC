/*
 mycontrollerで使用する関数
*/
#include <cuda.h>
__host__ __device__ double check_constraint(double u, double lower_c, double upper_c, double zeta);
__host__ __device__ double barrierConsraint(double obj, double lower_c, double upper_c, double sRho);