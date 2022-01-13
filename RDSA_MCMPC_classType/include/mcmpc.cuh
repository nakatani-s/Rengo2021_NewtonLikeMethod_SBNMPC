/*
This is header file, includes functions definitions, relates MCMPC for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
*/
#include <string.h>
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
#define InputSaturation
unsigned int countBlocks(unsigned int a, unsigned int b);
double calc_cost(double *inputSeq, double *cs, double *prm, double *ref, double *cnstrnt, double *we, IndexParams *Idx);
__host__ __device__ void compute_weighted_mean(double *out, IndexParams *Idx, int *indices, SampleInfo *info);
__global__ void devicePrnDebuggerSIF(double ts, SampleInfo *SIF);
__global__ void setup_RandomSeed(curandState *st, int seed);
__global__ void parallelSimForMC(double var, double *st, double *pr, double *re, double *co, double *we, double *mean, curandState *rndSeed, SampleInfo *info, IndexParams *Idx, double *cost_vec, int *indices);
__global__ void get_managed_indices(int *mng_indices_vec, int *thrust_indices_vec);
__global__ void calc_weighted_mean(double *out, IndexParams *Idx, int *indices, SampleInfo *info);
// __global__ void parallelSimForMCMPC( );
