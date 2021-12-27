/*
This is header file, includes functions definitions, relates MCMPC for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
*/

#include<cuda.h>
#include<curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
// #include "params.cuh"
#include "DataStructures.cuh"
#include "myController.cuh"
#include "integrator.cuh"
// #include "dynamics.cuh"

unsigned int countBlocks(unsigned int a, unsigned int b);
__global__ void setup_RandomSeed(curandState *st, int seed);
__global__ void parallelSimForMC(double var, double *st, double *pr, double *re, double *co, double *we, double *mean, curandState *rndSeed, SampleInfo *SIF, IndexParams *Idx, double *cost_vec);
// __global__ void parallelSimForMCMPC( );
