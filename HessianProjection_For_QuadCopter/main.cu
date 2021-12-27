/*
This file includes main function for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
＊高速化（正規方程式の係数行列の計算はcublasにお任せ）
＊gradient相当の係数はRDSAを使用
＊シミュレーションモデルは、クアッドコプター（外乱ありからの姿勢回復）
*/

#include "RDSA_NLMCMPC.cuh"

int main(int argc, char **argv)
{
    /* インデックスパラメータの取得 */
    IndexParams *gIdx, *devIdx;
    gIdx = (IndexParams*)malloc(sizeof(IndexParams));
    set_IdxParams( gIdx );
    const IndexParams *Idx = gIdx;
    CHECK(cudaMalloc(&devIdx, sizeof(IndexParams)));
    CHECK(cudaMemcpy(devIdx, gIdx, sizeof(IndexParams), cudaMemcpyHostToDevice));

    /* Get System & Controller Valiables*/
    SystemControlVariable *hostSCV, *deviceSCV;
    hostSCV = (SystemControlVariable*)malloc(sizeof(SystemControlVariable));
    init_variables( hostSCV );
    CHECK( cudaMalloc(&deviceSCV, sizeof(SystemControlVariable)) );
    CHECK( cudaMemcpy(deviceSCV, hostSCV, sizeof(SystemControlVariable), cudaMemcpyHostToDevice) );

    /* Setup GPU parameters */
    unsigned int numBlocks, randomNums;
    randomNums = NUM_OF_SAMPLES * (DIM_OF_INPUT + 1) * HORIZON;
    numBlocks = countBlocks(NUM_OF_SAMPLES, THREAD_PER_BLOCKS);
    /* Setup Random Seed for device */
    curandState *devRandSeed;
    cudaMalloc((void **)&devRandSeed, randomNums * sizeof(curandState));
    setup_kernel<<<NUM_OF_SAMPLES, (DIM_OF_INPUT + 1) * HORIZON>>>(devRandSeed, rand());
    cudaDeviceSynchronize();

    /* Setup Datastructure includes cost, InputSequences, ..., etc. */
    SampleInfo *devSampleInfo, *hostSampleInfo;
    hostSampleInfo = (SampleInfo *)malloc(sizeof(SampleInfo) * NUM_OF_SAMPLES);
    CHECK( cudaMalloc(&devSampleInfo, sizeof(SampleInfo) * NUM_OF_SAMPLES) );

    /* Matrix or Vector Array with result of Quadratic Hypersurface Regression */
    double *Hessian, *UpperHessian, *Gradient;
    CHECK( cudaMalloc(&Hessian, sizeof(double) * Idx->HessianSize) );
    CHECK( cudaMalloc(&UpperHessian, sizeof(double) * Idx->HessianSize) );
    CHECK( cudaMalloc(&Gradient, sizeof(double) * Idx->InputByHorizon) );

    /* For executing Least-Squares method by cublas */ 
    double *CoeMatrix, *bVector, *TensortX, *TransposeX, *TensortL;
    CHECK( cudaMalloc(&CoeMatrix, sizeof(double) * Idx->size_HessElements) );
    CHECK( cudaMalloc(&TensortX, sizeof(double) * Idx->sz_FittingSamples * Idx->sz_HessElements) );
    CHECK( cudaMalloc(&TransposeX, sizeof(double) * Idx->sz_FittingSamples * Idx->sz_HessElements) );
    CHECK( cudaMalloc(&TensortL, sizeof(double) * Idx->sz_FittingSamples) );

    /* DataStructures for Tensor Vector per Samples */
    QHP *devQHP; // Quadratic HyperSurface Parameters (QHP)
    CHECK( cudaMalloc(&devQHP, sizeof(QHP) * Idx->sz_FittingSamples) );

    /* Variables For cuSolver and cuBlas functions */
    /* GPGPUの行列演算ライブラリ用に宣言が必要なやつ */
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH), "Failed to Create cusolver handle" );
    CHECK_CUBLAS( cublasCreate(&cublasH), "Failed to create cublas handle" );

    /* For QR Decomposition  <---> QR分解用の変数、workspaceの確保 */
    double *ws_QR_operation = NULL;
    int geqrf_work_size = 0;
    int ormqr_work_size = 0;
    int QR_work_size = 0;
    const int m_Rmatrix = Idx->sz_HessElements;

    /* thrust使用のためのホスト/デバイス用のベクトルの宣言 */
    thrust::host_vector<int> indices_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<int> indices_dev_vec = indices_dev_vec;
    thrust::host_vector<double> sort_key_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<double> sort_key_dev_vec = sort_key_host_vec; 
    



}