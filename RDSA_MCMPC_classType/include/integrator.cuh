/*
    積分近似のメソッドを宣言
    *
    *
*/
#include <math.h>
#include <cuda.h>

#include "myController.cuh"

#ifndef INTEGRATOR_CUH
#define INTEGRATOR_CUH

__host__ __device__ void transition_Eular(double *st, double *dst, double delta, int dimState);

#endif