/*
This file includes all header files for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
＊高速化（正規方程式の係数行列の計算はcublasにお任せ）
＊gradient相当の係数はRDSAを使用
＊シミュレーションモデルは、クアッドコプター（外乱ありからの姿勢回復）
*/

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
// #include <iomanip>

#include "include/params.cuh"
#include "include/init.cuh"
#include "include/matrix.cuh"
#include "include/mcmpc.cuh"

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

/* For cublas definition */ 
cublasFillmode_t uplo = CUBLAS_FILL_MODE_LOWER;
cublasSideMode_t side = CUBLAS_SIDE_LEFT;
cublasOperation_t trans = CUBLAS_OP_T;
cublasOperation_t trans_N = CUBLAS_OP_N;
cublasFillMode_t uplo_QR = CUBLAS_FILL_MODE_UPPER;
cublasDiagType_t cub_diag = CUBLAS_DIAG_NON_UNIT;
const int nrhs = 1;
