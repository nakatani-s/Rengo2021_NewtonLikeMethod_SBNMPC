/*
 MCMPC.cuh 
*/

#include<cuda.h>
#include<curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"

void shift_Input_vec( double *inputVector);
void weighted_mean(double *Output, int num_elite, SampleInfo *hInfo);

__global__ void setup_kernel(curandState *state,int seed);
__global__ void getEliteSampleInfo( SampleInfo *Elite, SampleInfo *All, int *indices);
__global__ void MCMPC_Cart_and_SinglePole( SystemControlVariable *SCV, double var, curandState *randomSeed, double *mean, SampleInfo *Info, double *cost_vec);
