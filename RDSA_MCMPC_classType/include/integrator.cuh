/*
    積分近似のメソッドを宣言
    *
    *
*/
#include <math.h>
#include <string.h>
#include <cuda.h>

#include "myController.cuh"
#include "DataStructures.cuh"

#ifndef INTEGRATOR_CUH
#define INTEGRATOR_CUH

__host__ __device__ void transition_Eular(double *st, double *dst, double delta, int dimState);
__host__ __device__ void runge_kutta_45(double *st, int state_dim, double *input, double *prm, double t_delta);

#endif