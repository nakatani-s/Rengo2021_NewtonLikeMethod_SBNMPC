/*
This is header file, includes functions definitions, relates MCMPC for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
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

// #include "../rdsa_mcmpc_setupfile.cuh"
#include "myController.cuh"
#include "DataStructures.cuh"
#include "init.cuh"
#include "mcmpc.cuh"

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



class rdsa_mcmpc{
private:
    FILE *fp_state, *fp_input;
    int time_steps;

    IndexParams *gIdx, *devIdx;
    
    curandState *devRandSeed;

    SampleInfo *info;
    QHP *qhp;
    double *Hessian, *UpperHessian, *Gradient;

    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    cublasFillMode_t uplo;
    cublasSideMode_t side;
    cublasOperation_t trans;
    cublasOperation_t trans_N;
    cublasFillMode_t uplo_QR;
    cublasDiagType_t cub_diag;

    double *ws_QR_operation;
    int geqrf_work_size;
    int ormqr_work_size;
    int QR_work_size;
    int m_Rmatrix;
    int nrhs;
    double *QR_tau;
    double *hQR_tau;

    thrust::host_vector<int> indices_host_vec;
    thrust::device_vector<int> indices_device_vec;
    thrust::host_vector<double> sort_key_host_vec;
    thrust::device_vector<double> sort_key_device_vec;

public:
    rdsa_mcmpc(CoolingMethod method); //コンストラクタ
    ~rdsa_mcmpc(); //デストラクタ
    /*rdsa_mcmpc(const rdsa_mcmpc &) = delete;
    rdsa_mcmpc &operator=(const rdsa_mcmpc &) = delete;
    rdsa_mcmpc(rdsa_mcmpc &&) = delete;
    rdsa_mcmpc &operator=(rdsa_mcmpc &&) = delete;*/

    double costValue;
    /* 推定入力値の一時保存用の配列 */
    CoolingMethod cMethod;
    double *hostDataMC, *hostDataRDSA, *deviceDataMC, *deviceDataRDSA;

    // SystemControllerVariable *hstSCV, *devSCV;
    double *_state, *_reference,*_parameters, *_constraints, *_weightMatrix;
    double *CoeMatrix, *bVector, *TensortX, *TransposeX, *TensortL;
    // double *hstPrm;

    unsigned int numBlocks, randomNums, threadPerBlocks;

    // static rdsa_mcmpc &get_instance();
    void execute_rdsa_mcmpc(double *CurrentInput);
    void set(double *a, valueType type);
    // static void set_controller_param(double *prm);
    void do_forward_simulation(double *state, double *input, IntegralMethod method);
    void write_data_to_file(double *input);

};

