#include<iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>

#include "include/params.cuh"
#include "include/init.cuh"
#include "include/Matrix.cuh"
#include "include/DataStructure.cuh"
#include "include/MCMPC.cuh"
#include "include/NewtonLikeMethod.cuh"
#include "include/optimum_conditions.cuh"
#include "include/dataToFile.cuh"
// #include "include/cudaErrorCheck.cuh"

#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                        \
    {                                                                \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                \
        printf("code:%d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                          \
        exit(1);                                                     \
    }                                                                \
}
#define CHECK_CUBLAS(call,str)                                                        \
{                                                                                     \
    if ( call != CUBLAS_STATUS_SUCCESS)                                               \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}
#define CHECK_CUSOLVER(call,str)                                                      \
{                                                                                     \
    if ( call != CUSOLVER_STATUS_SUCCESS)                                             \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}
