/*
-----------2021.12.21 makin start 
*/

#include "../include/myControllerSrc.cuh"

__host__ __device__ double check_constraint(double u, double lower_c, double upper_c, double zeta)
{
    double ret = u;
    if(u < lower_c)
    {
        ret = lower_c + zeta;
    }
    if(u > upper_c)
    {
        ret = upper_c - zeta;
    }

    return ret;
}

__host__ __device__ double barrierConsraint(double obj, double lower_c, double upper_c, double sRho)
{
    double ret = 0.0;
    ret += -logf(obj - lower_c)-logf(upper_c - obj);
    ret += sRho * (upper_c - lower_c);
    return ret;
}