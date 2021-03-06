/*
*/
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


int main(int argc, char **argv)
{
    /* ?????????????????????????????????????????????????????? */
    cusolverDnHandle_t cusolverH = NULL;
    // cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH),"Failed to Create cusolver handle");

    cublasHandle_t handle_cublas = 0;
    cublasCreate(&handle_cublas);

    /* ??????????????????????????????????????????????????????????????????*/
    FILE *fp_input, *fp_state, *opco;
    time_t timeValue;
    struct tm *timeObject;
    time( &timeValue );
    timeObject = localtime( &timeValue );
    char filename1[35], filename2[40], filename3[40];
    sprintf(filename1,"data_input_%d%d_%d%d.txt",timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour,timeObject->tm_min);
    sprintf(filename3,"data_state_%d%d_%d%d.txt",timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour,timeObject->tm_min);
    sprintf(filename2,"optimum_condition_%d%d_%d%d.txt", timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min);
    fp_input = fopen(filename1,"w");
    fp_state = fopen(filename3,"w");
    opco = fopen(filename2,"w");

    /* ?????????????????????????????????????????????????????????????????? */
    // double hostParams[DIM_OF_PARAMETERS], hostState[DIM_OF_STATES], hostConstraint[NUM_OF_CONSTRAINTS], hostWeightMatrix[DIM_OF_WEIGHT_MATRIX];
    SystemControlVariable *hostSCV, *deviceSCV;
    hostSCV = (SystemControlVariable*)malloc(sizeof(SystemControlVariable));
    init_variables( hostSCV );
    CHECK( cudaMalloc(&deviceSCV, sizeof(SystemControlVariable)) );
    CHECK( cudaMemcpy(deviceSCV, hostSCV, sizeof(SystemControlVariable), cudaMemcpyHostToDevice) );
    

    /* GPU??????????????????????????? */
    unsigned int numBlocks, /*randomBlocks,*/ randomNums; /*Blocks, dimHessian, numUnknownParamQHP, numUnknownParamHessian;*/
    randomNums = NUM_OF_SAMPLES * (DIM_OF_INPUT + 1) * HORIZON;
    // randomBlocks = countBlocks(randomNums, THREAD_PER_BLOCKS);
    numBlocks = countBlocks(NUM_OF_SAMPLES, THREAD_PER_BLOCKS);

    /* Fitting????????????????????????????????????????????? */
    IndexParams *gIdx, *devIdx;
    gIdx = (IndexParams*)malloc(sizeof(IndexParams));
    set_IndexParams( gIdx ); 
    const IndexParams *Idx = gIdx;
    CHECK(cudaMalloc(&devIdx, sizeof(IndexParams)));
    CHECK(cudaMemcpy(devIdx, gIdx, sizeof(IndexParams), cudaMemcpyHostToDevice));
    /*unsigned int paramsSizeQuadHyperPlane;
    const int InputByHorizon = HORIZON * DIM_OF_INPUT;
    const int PartIByH = PART_HORIZON * DIM_OF_INPUT; 
    randomNums = NUM_OF_SAMPLES * (DIM_OF_INPUT + 1) * HORIZON;
    // randomBlocks = countBlocks(randomNums, THREAD_PER_BLOCKS);
    numBlocks = countBlocks(NUM_OF_SAMPLES, THREAD_PER_BLOCKS);
    // Blocks = numBlocks;
    dimHessian = InputByHorizon * InputByHorizon;

    numUnknownParamQHP = NUM_OF_PARABOLOID_COEFFICIENT;
    numUnknownParamHessian = numUnknownParamQHP - (InputByHorizon + 1);
    paramsSizeQuadHyperPlane = numUnknownParamQHP; //?????????????????????????????????????????????????????????????????????????????????
    paramsSizeQuadHyperPlane = paramsSizeQuadHyperPlane + addTermForLSM;*/

    dim3 block_S(MAX_DIVISOR_S);
    dim3 grid_S((Idx->num_UKPrm_QC_S + block_S.x -1)/ block_S.x, Idx->num_UKPrm_QC_S);
    
    dim3 block_L(MAX_DIVISOR);
    // dim3 block(1,1);
    dim3 grid_L((Idx->num_UKPrm_QC_L + block_L.x - 1)/ block_L.x, Idx->num_UKPrm_QC_L);
    printf("#NumBlocks = %d\n", numBlocks);
    printf("grid_S(%d,%d) block_S(%d,%d)\n",grid_S.x, grid_S.y, block_S.x, block_S.y);
    printf("grid_L(%d,%d) block_L(%d,%d)\n", grid_L.x, grid_L.y, block_L.x, block_L.y);
    // printf("#NumBlocks = %d\n", numUnknownParamQHP);

#ifdef WRITE_MATRIX_INFORMATION
    double *WriteHessian, *WriteRegular;
    WriteHessian = (double *)malloc(sizeof(double)*Idx->dim_H_L);
    WriteRegular = (double *)malloc(sizeof(double)*Idx->pow_nUKPrm_QC_S);
    int timerParam[5] = { };
    dataName *name;
    name = (dataName*)malloc(sizeof(dataName)*5);
#endif

#ifdef READ_LQR_MATRIX
    double *part_of_CVector, *d_CVectorFromLQR;
    part_of_CVector = (double *)malloc(sizeof(double)*Idx->num_UKPrm_H_L);
    CHECK(cudaMalloc(&d_CVectorFromLQR, sizeof(double) * Idx->num_UKPrm_H_L));
    sprintf(name[2].inputfile, "HessianFromLQR.csv");
    name[2].dimSize = Idx->num_UKPrm_H_L;
    resd_InitSolution_Input(part_of_CVector, &name[2]);
    CHECK(cudaMemcpy(d_CVectorFromLQR, part_of_CVector, sizeof(double) * Idx->num_UKPrm_H_L, cudaMemcpyHostToDevice));
#endif

    /* MCMPC????????????????????????seed??????????????? */
    curandState *deviceRandomSeed;
    cudaMalloc((void **)&deviceRandomSeed, randomNums * sizeof(curandState));
    setup_kernel<<<NUM_OF_SAMPLES, (DIM_OF_INPUT + 1) * HORIZON>>>(deviceRandomSeed, rand());
    cudaDeviceSynchronize();
    
    /* ????????????????????????????????????????????????????????????????????????????????? */
    SampleInfo *deviceSampleInfo, *hostSampleInfo, *hostEliteSampleInfo, *deviceEliteSampleInfo;
    hostSampleInfo = (SampleInfo *)malloc(sizeof(SampleInfo) * NUM_OF_SAMPLES);
    hostEliteSampleInfo = (SampleInfo*)malloc(sizeof(SampleInfo) * NUM_OF_ELITES);
    cudaMalloc(&deviceSampleInfo, sizeof(SampleInfo) * NUM_OF_SAMPLES);
    cudaMalloc(&deviceEliteSampleInfo, sizeof(SampleInfo) * NUM_OF_ELITES);
    /*SampleInfo *TemporarySampleInfo, *deviceTempSampleInfo;
    TemporarySampleInfo = (SampleInfo *)malloc(sizeof(SampleInfo) * paramsSizeQuadHyperPlane);
    CHECK(cudaMalloc(&deviceTempSampleInfo, sizeof(SampleInfo) * paramsSizeQuadHyperPlane) );*/

    Tolerance *hostTol;
    hostTol = (Tolerance*)malloc(sizeof(Tolerance)*HORIZON+1);

    /* ???????????????????????????????????????????????????????????????????????????????????????????????????<---??????????????????????????????*/
    double *Hessian, *lowerHessian;
    double *Gradient;
    /* 1st ?????????????????????????????????????????????????????????????????????Malloc <-- ???cudaFree after 1st Estimation */ 
    CHECK( cudaMalloc(&Hessian, sizeof(double) * Idx->dim_H_L));
    CHECK( cudaMalloc(&lowerHessian, sizeof(double) * Idx->dim_H_L));
    CHECK( cudaMalloc(&Gradient, sizeof(double) * Idx->InputByHorizonL));

    /* ?????????????????????????????????????????????????????????????????????????????? */
    double *PartGmatrix, *PartCVector;
    double *Gmatrix, *CVector;
    double *PartTensortA, *PartTensortB, *PartTensortL;

    /*  2nd ?????????????????????????????????????????????????????????????????????????????????????????????2????????????????????????????????? */ 
    CHECK( cudaMalloc(&PartCVector, sizeof(double) * Idx->num_UKPrm_QC_S ));
    CHECK( cudaMalloc(&PartTensortA, sizeof(double) * Idx->num_UKPrm_QC_S * Idx->sz_LCLsamples_S));
    CHECK( cudaMalloc(&PartTensortB, sizeof(double) * Idx->num_UKPrm_QC_S * Idx->sz_LCLsamples_S));
    CHECK( cudaMalloc(&PartTensortL, sizeof(double) * Idx->sz_LCLsamples_S) );
    CHECK( cudaMalloc(&PartGmatrix, sizeof(double) * Idx->pow_nUKPrm_QC_S));

    /* 1st??????????????????????????????????????????????????????????????????Malloc <-- ???cudaFree after 1st Estimation ended */ 
    CHECK( cudaMalloc(&CVector, sizeof(double) * Idx->num_UKPrm_QC_L) );
    CHECK( cudaMalloc(&Gmatrix, sizeof(double) * Idx->pow_nUKPrm_QC_L));


    QHP *deviceQHP, *devicePartQHP;
    CHECK( cudaMalloc(&devicePartQHP, sizeof(QHP) * Idx->sz_LCLsamples_S) );
    // CHECK( cudaMalloc(&deviceQHP, sizeof(QHP) * paramsSizeQuadHyperPlane) );
    CHECK( cudaMalloc(&deviceQHP, sizeof(QHP) * Idx->sz_LCLsamples_L) ); // 1st step?????????Free??????

    unsigned int qhpBlocks_L, qhpBlocks_S;
    qhpBlocks_S = countBlocks(Idx->sz_LCLsamples_S, THREAD_PER_BLOCKS);
    qhpBlocks_L = countBlocks(Idx->sz_LCLsamples_L, THREAD_PER_BLOCKS);
    printf("#qhpBlocksL := %d || qhpBlocksS := %d\n", qhpBlocks_L, qhpBlocks_S);

    // ????????????????????????????????????????????????????????????
    const int m_Rmatrix = Idx->num_UKPrm_QC_L;
    const int mS_Rmatrix = Idx->num_UKPrm_QC_S;

    int work_size, w_si_hessian;
    double *work_space, *w_sp_hessian;
    int *devInfo;
    CHECK( cudaMalloc((void**)&devInfo, sizeof(int) ) );

    /* thrust???????????????????????????/???????????????????????????????????? */ 
    thrust::host_vector<int> indices_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<int> indices_device_vec = indices_host_vec;
    thrust::host_vector<double> sort_key_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<double> sort_key_device_vec = sort_key_host_vec; 

    /* ???????????????????????????????????????????????????*/
    double *hostData, *deviceData, *hostTempData, *deviceTempData;
    hostData = (double *)malloc(sizeof(double) * Idx->InputByHorizonL);
    hostTempData = (double *)malloc(sizeof(double) * Idx->InputByHorizonL);
    CHECK(cudaMalloc(&deviceData, sizeof(double) * Idx->InputByHorizonL));
    CHECK(cudaMalloc(&deviceTempData, sizeof(double) * Idx->InputByHorizonL));
    for(int i = 0; i < Idx->InputByHorizonL; i++){
        if(i % 4 == 0){
            hostData[i] = hostSCV->params[0];
        }
    }
    hostData[0] = hostSCV->params[0];
    printf("u[0] = %lf, u[4] = %lf, u[8] = %lf", hostData[0], hostData[4], hostData[8]);
    CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * Idx->InputByHorizonL, cudaMemcpyHostToDevice));

    /* ???????????????????????? */
    double F_input[4] = { };
    double MCMPC_F[4] = {};
    double Proposed_F[4] = {};
    // double costFromMCMPC, costFromProposed, toleranceFromMCMPC, toleranceFromProposed;
    double cost_now;
    double COST_MC[2] = { }; // COST COST+LogBarrierTerm
    double COST_NLM[2] = { }; // COST COST+LogBarrierTerm
    // double optimumConditions[2] = { };
    // double optimumCondition_p[2] = { };
    double var;

    float process_gpu_time, procedure_all_time;
    float diff_time;
    clock_t start_t, stop_t;
    clock_t tensor_start_t, tensor_end_t, matcalc_s_t, matcalc_end_t;
    cudaEvent_t start, stop;

#ifdef USING_QR_DECOMPOSITION
    // double *QR_work_space = NULL;
    double *ws_QR_operation = NULL;
    double *ws_blc_QR_operation = NULL;
    int geqrf_work_size = 0;
    int ormqr_work_size = 0;
    int QR_work_size = 0;
    const int nrhs = 1;
    double *QR_tau = NULL;
    double *hQR_tau = NULL;
    double *blc_QR_tau = NULL;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasOperation_t trans_N = CUBLAS_OP_N;
    cublasFillMode_t uplo_QR = CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t cub_diag = CUBLAS_DIAG_NON_UNIT;
    CHECK(cudaMalloc((void**)&QR_tau, sizeof(double) * Idx->num_UKPrm_QC_L));
    CHECK(cudaMalloc((void**)&hQR_tau, sizeof(double) * Idx->InputByHorizonL));
    CHECK(cudaMalloc((void**)&blc_QR_tau, sizeof(double) * Idx->num_UKPrm_QC_S));
#endif

    mInputSystem tsDims = MultiInput;
    for(int t = 0; t < SIM_TIME; t++)
    {
        // shift_Input_vec( hostData );
        CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * Idx->InputByHorizonL, cudaMemcpyHostToDevice) );
        start_t = clock();

        if(t == 0)
        {
            start_t = clock();
            for(int iter = 0; iter < ITERATIONS_MAX; iter++)
            {
                var = variance / sqrt(iter + 1);
                // var = variance / 2;
                // MCMPC_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                // MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                MCMPC_QuaternionBased_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                cudaDeviceSynchronize();
                thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());

                
                getEliteSampleInfo_multiInput<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                // getEliteSampleInfo<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                CHECK( cudaMemcpy(hostEliteSampleInfo, deviceEliteSampleInfo, sizeof(SampleInfo) * NUM_OF_ELITES, cudaMemcpyDeviceToHost) );
                // weighted_mean(hostData, NUM_OF_ELITES, hostSampleInfo);
                weighted_mean_multiInput(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
                // weighted_mean_multiInput_Quadrotor(hostData, NUM_OF_ELITES, hostEliteSampleInfo, hostSCV);
                // weighted_mean(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
                for(int uIndex = 0; uIndex < DIM_OF_INPUT; uIndex++)
                {
                    MCMPC_F[uIndex] = hostData[uIndex];
                }
                /*if(iter == 0)
                {
                    sprintf(name[2].inputfile, "initSolution.txt");
                    name[2].dimSize = HORIZON;
                    resd_InitSolution_Input(hostData, &name[2]);
                }*/
                CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * Idx->InputByHorizonL, cudaMemcpyHostToDevice) );

                /* get 1st Step's Full dimentional Hessian */
                /* ???????????????????????????????????????????????????????????????????????????????????????????????? */ 
                /* ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????LQR?????????????????? */
                if(iter == ITERATIONS_MAX -1)
                {
                    var = neighborVar;
                    MCMPC_QuaternionBased_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                    // MCMPC_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                    // MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                    cudaDeviceSynchronize();
                    thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                    thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());

                    NewtonLikeMethodGetTensorVectorTest<<< qhpBlocks_L, THREAD_PER_BLOCKS>>>(deviceQHP, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ),tsDims);
                    cudaDeviceSynchronize();
                    if(Idx->num_UKPrm_QC_L < 1024){
                        NewtonLikeMethodGetRegularMatrix<<<Idx->num_UKPrm_QC_L, Idx->num_UKPrm_QC_L>>>(Gmatrix, deviceQHP, Idx->sz_LCLsamples_L);
                    }else{
                        NewtonLikeMethodGetRegularMatrixTypeB<<<grid_L, block_L>>>(Gmatrix, deviceQHP, Idx->sz_LCLsamples_L, Idx->num_UKPrm_QC_L);
                        cudaDeviceSynchronize();
                    }
                    NewtonLikeMethodGetRegularVector<<<Idx->num_UKPrm_QC_L, 1>>>(CVector, deviceQHP, Idx->sz_LCLsamples_L);
                    cudaDeviceSynchronize();
                    CHECK_CUSOLVER( cusolverDnDgeqrf_bufferSize(cusolverH, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                    // CHECK_CUSOLVER( cusolverDnSormqr_bufferSize(cusolverH, side, trans, m_Rmatrix, nrhs, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, CVector, m_Rmatrix, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                    CHECK_CUSOLVER( cusolverDnDormqr_bufferSize(cusolverH, side, trans, m_Rmatrix, nrhs, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, CVector, m_Rmatrix, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                    QR_work_size = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
                    CHECK( cudaMalloc((void**)&ws_QR_operation, sizeof(double) * QR_work_size) );

                    CHECK_CUSOLVER( cusolverDnDgeqrf(cusolverH, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, ws_QR_operation, QR_work_size, devInfo),"Failed to compute QR factorization" );
                    CHECK_CUSOLVER( cusolverDnDormqr(cusolverH, side, trans, m_Rmatrix, nrhs, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, CVector, m_Rmatrix, ws_QR_operation, QR_work_size, devInfo), "Failed to compute Q^T*B" );
                    CHECK(cudaDeviceSynchronize());
                    matcalc_s_t = clock();
                    CHECK_CUBLAS( cublasDtrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, m_Rmatrix, nrhs, &alpha, Gmatrix, m_Rmatrix, CVector, m_Rmatrix), "Failed to compute X = R^-1Q^T*B" );
                    CHECK(cudaDeviceSynchronize());
                    matcalc_end_t = clock();
                    diff_time = matcalc_end_t-matcalc_s_t;
                    printf("matrix making time := %lf\n", diff_time / CLOCKS_PER_SEC );
                    sleep(1);
                    // CHECK(cudaDeviceSynchronize());
                    NewtonLikeMethodGetHessianOriginal<<<Idx->InputByHorizonL, Idx->InputByHorizonL>>>(Hessian, CVector);
                    CHECK(cudaDeviceSynchronize());
#ifdef WRITE_MATRIX_INFORMATION     
                    get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t+1200);
                    sprintf(name[1].name, "HessianMatrix");
                    name[1].dimSize = Idx->InputByHorizonL;
                    CHECK(cudaMemcpy(WriteHessian, Hessian, sizeof(double) * Idx->InputByHorizonL * Idx->InputByHorizonL, cudaMemcpyDeviceToHost));
                    write_Matrix_Information(WriteHessian, &name[1], timerParam);
#endif
                    NewtonLikeMethodGetLowerTriangle<<<Idx->InputByHorizonL, Idx->InputByHorizonL>>>(lowerHessian, Hessian);
                    CHECK(cudaDeviceSynchronize());
#ifdef WRITE_MATRIX_INFORMATION     
                    get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t+1201);
                    sprintf(name[1].name, "HessianMatrix");
                    name[1].dimSize = Idx->InputByHorizonL;
                    CHECK(cudaMemcpy(WriteHessian, lowerHessian, sizeof(double) * Idx->InputByHorizonL * Idx->InputByHorizonL, cudaMemcpyDeviceToHost));
                    write_Matrix_Information(WriteHessian, &name[1], timerParam);
#endif
                    // NewtonLikeMethodGetFullHessianLtoU<<<HORIZON, HORIZON>>>(Hessian, lowerHessian);
                    NewtonLikeMethodGetFullHessianUtoL<<<Idx->InputByHorizonL, Idx->InputByHorizonL>>>(lowerHessian, Hessian);
                    NewtonLikeMethodGetGradient<<<Idx->InputByHorizonL, 1>>>(Gradient, CVector, Idx->num_UKPrm_H_L);
                    MatrixMultiplyOperation<<<Idx->InputByHorizonL,Idx->InputByHorizonL>>>(Hessian, 2.0, lowerHessian);
#ifdef WRITE_MATRIX_INFORMATION     
                    get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t);
                    sprintf(name[1].name, "HessianMatrix");
                    name[1].dimSize = Idx->InputByHorizonL;
                    CHECK(cudaMemcpy(WriteHessian, Hessian, sizeof(double) * Idx->InputByHorizonL * Idx->InputByHorizonL, cudaMemcpyDeviceToHost));
                    write_Matrix_Information(WriteHessian, &name[1], timerParam);
#endif

                    CHECK_CUSOLVER( cusolverDnDgeqrf_bufferSize(cusolverH, Idx->InputByHorizonL, Idx->InputByHorizonL, Hessian, Idx->InputByHorizonL, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                    CHECK_CUSOLVER( cusolverDnDormqr_bufferSize(cusolverH, side, trans, Idx->InputByHorizonL, nrhs, Idx->InputByHorizonL, Hessian, Idx->InputByHorizonL, hQR_tau, Gradient, Idx->InputByHorizonL, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                    w_si_hessian = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
                    CHECK( cudaMalloc((void**)&w_sp_hessian, sizeof(double) * w_si_hessian) );

                    /*QR decomposition */ 
                    CHECK_CUSOLVER( cusolverDnDgeqrf(cusolverH, Idx->InputByHorizonL, Idx->InputByHorizonL, Hessian, Idx->InputByHorizonL, hQR_tau, w_sp_hessian, w_si_hessian, devInfo),"Failed to compute QR factorization" );
                    CHECK_CUSOLVER( cusolverDnDormqr(cusolverH, side, trans, Idx->InputByHorizonL, nrhs, Idx->InputByHorizonL, Hessian, Idx->InputByHorizonL, hQR_tau, Gradient, Idx->InputByHorizonL, w_sp_hessian, w_si_hessian, devInfo), "Failed to compute Q^T*B" );
                    CHECK(cudaDeviceSynchronize());
                    CHECK_CUBLAS( cublasDtrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, Idx->InputByHorizonL, nrhs, &m_alpha, Hessian, Idx->InputByHorizonL, Gradient, Idx->InputByHorizonL), "Failed to compute X = R^-1Q^T*B" );
                    CHECK(cudaDeviceSynchronize());
                    NewtonLikeMethodCopyVector<<<Idx->InputByHorizonL, 1>>>(deviceTempData, Gradient);
                    CHECK(cudaDeviceSynchronize());

                    cudaFree(Gmatrix);
                    cudaFree(deviceQHP);
                    cudaFree(QR_tau);
                    cudaFree(ws_QR_operation);
                    CHECK( cudaMemcpy(hostTempData, deviceTempData, sizeof(double) * Idx->InputByHorizonL, cudaMemcpyDeviceToHost) );
                    CHECK(cudaDeviceSynchronize());
                    // NewtonLikeMethodInputSaturation(hostTempData, hostSCV->constraints[1], hostSCV->constraints[0]);
                    // Quadrotor_Input_recalculation(hostTempData, hostSCV);
                    Quadrotor_InputSaturation(hostTempData, hostSCV);
                    // cudaFree(hQR_tau);
                    // invGmatrix, ansCVector?????????????????????????????????????????????????????????????????????????????????QR?????????????????????????????????????????????????????????
                }
                // calc_Cost_Quadrotor(COST_MC, hostData, hostSCV);
                calc_Cost_QuaternionBased_Quadrotor(COST_MC, hostData, hostSCV);
                calc_Cost_QuaternionBased_Quadrotor(COST_NLM, hostTempData, hostSCV); 
                // calc_OC_for_Cart_and_SinglePole_hostF(optimumConditions, hostData, hostSCV, hostTol);
                // printf("cost :: %lf   KKT_Error :: %lf\n", optimumConditions[0], optimumConditions[1]);
#ifdef READ_LQR_MATRIX
                NewtonLikeMethodCopyVector<<<Idx->num_UKPrm_H_L,1>>>(CVector, d_CVectorFromLQR);
                CHECK(cudaDeviceSynchronize());
#endif
            }
            /*name[1].dimSize = HORIZON;
            sprintf(name[1].name,"InitInputData.txt");
            write_Vector_Information(hostData, &name[1]);*/
            stop_t = clock();
            procedure_all_time = stop_t - start_t;
            printf("Geometrical cooling MCMPC computation time :: %lf\n", procedure_all_time / CLOCKS_PER_SEC);
        }else{
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            start_t = clock();
            for(int iter = 0; iter < ITERATIONS; iter++)
            {
                var = variance / 1.0;
                var = var / sqrt(iter + 1);
                // MCMPC_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                // MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                MCMPC_QuaternionBased_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                cudaDeviceSynchronize();
                thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());
            
                // CHECK( cudaMemcpy(hostSampleInfo, deviceSampleInfo, sizeof(SampleInfo) * NUM_OF_SAMPLES, cudaMemcpyDeviceToHost) );
                getEliteSampleInfo_multiInput<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                // getEliteSampleInfo<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                CHECK( cudaMemcpy(hostEliteSampleInfo, deviceEliteSampleInfo, sizeof(SampleInfo) * NUM_OF_ELITES, cudaMemcpyDeviceToHost) );
                // weighted_mean_multiInput_Quadrotor(hostData, NUM_OF_ELITES, hostEliteSampleInfo, hostSCV);
                weighted_mean_multiInput(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
                // IT_weighted_mean_multiInput(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
                // weighted_mean(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
                for(int uIndex = 0; uIndex < DIM_OF_INPUT; uIndex++)
                {
                    MCMPC_F[uIndex] = hostData[uIndex];
                }
                
                CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * Idx->InputByHorizonL, cudaMemcpyHostToDevice) );
                if(iter == ITERATIONS -1){
                    var = neighborVar;
                    MCMPC_QuaternionBased_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                    // MCMPC_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                    // MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                    cudaDeviceSynchronize();
                    thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                    thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());

                    
                    // 2021.10.6????????????
                    // ?????????TensorVector????????????????????????L(x,u)- constant_L(x,u,a)?????????????????????
                    // ???????????????????????????N??N?????????index????????????tensorVector????????????????????????CVector???InputSeq??????L(x,u,a)???????????????
                    // ?????????????????????????????????????????????????????????????????????????????????differL??????????????????DataStructures??????????????????
                    // NewtonLikeMethodGetTensorVectorTest<<< qhpBlocks, THREAD_PER_BLOCKS>>>(deviceQHP, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ),tsDims);
                    // NewtonLikeMethodGetTensorVectorPartial<<< qhpBlocks_S, THREAD_PER_BLOCKS>>>(devicePartQHP, deviceSampleInfo, CVector, thrust::raw_pointer_cast( indices_device_vec.data() ), devIdx,tsDims);
                    NewtonLikeMethodGetTensorVectorPartialDirect<<< qhpBlocks_S, THREAD_PER_BLOCKS>>>(devicePartQHP, PartTensortA, PartTensortB, deviceSampleInfo, CVector, thrust::raw_pointer_cast( indices_device_vec.data() ), devIdx,tsDims);
                    cudaDeviceSynchronize();
                    NewtonLikeMethodGetTensorVectorNoIndex<<< qhpBlocks_S, THREAD_PER_BLOCKS>>>(PartTensortL, devicePartQHP, devIdx);
                    
                    tensor_start_t = clock();
                    if(Idx->num_UKPrm_QC_S < 1024){
                        // transpose ???????????????????????????????????????
                        CHECK_CUBLAS( cublasDgemm(handle_cublas, trans_N, trans, Idx->num_UKPrm_QC_S, Idx->num_UKPrm_QC_S, Idx->sz_LCLsamples_S, &alpha, PartTensortA, Idx->num_UKPrm_QC_S, PartTensortB, Idx->num_UKPrm_QC_S, &beta, PartGmatrix, Idx->num_UKPrm_QC_S), "TensorOperation Failed");
                        // NewtonLikeMethodGetRegularMatrix<<<Idx->num_UKPrm_QC_S, Idx->num_UKPrm_QC_S>>>(PartGmatrix, devicePartQHP, Idx->sz_LCLsamples_S);
                        // NewtonLikeMethodGetRegularMatrixTypeB<<<grid_S, block_S>>>(PartGmatrix, devicePartQHP, Idx->sz_LCLsamples_S, Idx->num_UKPrm_QC_S);
                        cudaDeviceSynchronize();
                    }else{
                        CHECK_CUBLAS( cublasDgemm(handle_cublas, trans_N, trans, Idx->num_UKPrm_QC_S, Idx->num_UKPrm_QC_S, Idx->sz_LCLsamples_S, &alpha, PartTensortA, Idx->num_UKPrm_QC_S, PartTensortB, Idx->num_UKPrm_QC_S, &beta, PartGmatrix, Idx->num_UKPrm_QC_S), "TensorOperation Failed");
                        // NewtonLikeMethodGetRegularMatrixTypeB<<<grid_S, block_S>>>(PartGmatrix, devicePartQHP, Idx->sz_LCLsamples_S, Idx->num_UKPrm_QC_S);
                        cudaDeviceSynchronize();
                    }
                    
                    tensor_end_t = clock();
                    diff_time = tensor_end_t - tensor_start_t;
                    printf("tensor time := %lf ", diff_time / CLOCKS_PER_SEC );
                    // NewtonLikeMethodGetRegularVectorPartial<<<Idx->num_UKPrm_QC_S, 1>>>(PartCVector, devicePartQHP, Idx->sz_LCLsamples_S);
                    // CHECK_CUBLAS( cublasDtrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, Idx->InputByHorizonL, nrhs, &m_alpha, Hessian, Idx->InputByHorizonL, Gradient, Idx->InputByHorizonL), "Failed to compute X = R^-1Q^T*B" );
                    CHECK_CUBLAS( cublasDgemm(handle_cublas, trans_N, trans, Idx->num_UKPrm_QC_S, 1, Idx->sz_LCLsamples_S, &alpha, PartTensortA, Idx->num_UKPrm_QC_S, PartTensortL, 1, &beta, PartCVector, Idx->num_UKPrm_QC_S), "TensorOperation Failed");
                    // printf("hoge %d hoge\n",t);
                    cudaDeviceSynchronize();

                    // NewtonLikeMethodCopyTensorVector<<<grid, block>>>(Gmatrix, deviceQHP, NUM_OF_PARABOLOID_COEFFICIENT);
#ifdef WRITE_MATRIX_INFORMATION
                    if(t<100){
                        if(t % 1 == 0){
                            get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t);
                            sprintf(name[0].name, "RegularMatrix");
                            name[0].dimSize = Idx->num_UKPrm_QC_S;
                            CHECK(cudaMemcpy(WriteRegular, PartGmatrix, sizeof(double) * Idx->pow_nUKPrm_QC_S, cudaMemcpyDeviceToHost));
                            write_Matrix_Information(WriteRegular, &name[0], timerParam);
                        }
                    }
#endif
                    matcalc_s_t = clock();
                    if(t==1){
                        CHECK_CUSOLVER( cusolverDnDgeqrf_bufferSize(cusolverH, mS_Rmatrix, mS_Rmatrix, PartGmatrix, mS_Rmatrix, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                        CHECK_CUSOLVER( cusolverDnDormqr_bufferSize(cusolverH, side, trans, mS_Rmatrix, nrhs, mS_Rmatrix, PartGmatrix, mS_Rmatrix, blc_QR_tau, PartCVector, mS_Rmatrix, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                        QR_work_size = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
                        /*-----Error Detect -----*/
                        printf("index == %d\n",QR_work_size);
                        CHECK( cudaMalloc((void**)&ws_blc_QR_operation, sizeof(double) * QR_work_size) );
                    }
                    /* compute QR factorization */ 
                    CHECK_CUSOLVER( cusolverDnDgeqrf(cusolverH, mS_Rmatrix, mS_Rmatrix, PartGmatrix, mS_Rmatrix, blc_QR_tau, ws_blc_QR_operation, QR_work_size, devInfo),"Failed to compute QR factorization" );
                    
                    CHECK_CUSOLVER( cusolverDnDormqr(cusolverH, side, trans, mS_Rmatrix, nrhs, mS_Rmatrix, PartGmatrix, mS_Rmatrix, blc_QR_tau, PartCVector, mS_Rmatrix, ws_blc_QR_operation, QR_work_size, devInfo), "Failed to compute Q^T*B" );
                    CHECK(cudaDeviceSynchronize());
                    
                    CHECK_CUBLAS( cublasDtrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, mS_Rmatrix, nrhs, &alpha, PartGmatrix, mS_Rmatrix, PartCVector, mS_Rmatrix), "Failed to compute X = R^-1Q^T*B" );
                    CHECK(cudaDeviceSynchronize());
                    matcalc_end_t = clock();
                    diff_time = matcalc_end_t-matcalc_s_t;
                    printf("matrix making time := %lf\n", diff_time / CLOCKS_PER_SEC );
                    
                    CHECK(cudaDeviceSynchronize());
                    matcalc_s_t = clock();
                    // ???????????????????????????????????????????????????
                    NewtonLikeMethodGetBLCHessian<<<Idx->InputByHorizonL, Idx->InputByHorizonL>>>(Hessian, PartCVector, CVector, devIdx);
                    CHECK(cudaDeviceSynchronize());

                    NewtonLikeMethodGetLowerTriangle<<<Idx->InputByHorizonL, Idx->InputByHorizonL>>>(lowerHessian, Hessian);
                    CHECK(cudaDeviceSynchronize());
                    // NewtonLikeMethodGetFullHessianLtoU<<<HORIZON, HORIZON>>>(Hessian, lowerHessian);
                    NewtonLikeMethodGetFullHessianUtoL<<<Idx->InputByHorizonL, Idx->InputByHorizonL>>>(lowerHessian, Hessian);
                    NewtonLikeMethodGetGradient<<<Idx->InputByHorizonL, 1>>>(Gradient, PartCVector, Idx->num_UKPrm_H_S);
                    MatrixMultiplyOperation<<<Idx->InputByHorizonL,Idx->InputByHorizonL>>>(Hessian, 2.0, lowerHessian);
                    matcalc_end_t = clock();
                    diff_time = matcalc_end_t-matcalc_s_t;
                    printf("Hessian making time := %lf\n", diff_time / CLOCKS_PER_SEC );
#ifdef WRITE_MATRIX_INFORMATION
                    if(t<20){
                        if(t % 1 == 0){
                            get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t);
                            sprintf(name[1].name, "HessianMatrix");
                            name[1].dimSize = Idx->InputByHorizonL;
                            CHECK(cudaMemcpy(WriteHessian, Hessian, sizeof(double) * Idx->InputByHorizonL * Idx->InputByHorizonL, cudaMemcpyDeviceToHost));
                            write_Matrix_Information(WriteHessian, &name[1], timerParam);
                        }
                    }

                    if(700<t && t<750){
                        if(t % 1 == 0){
                            get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t);
                            sprintf(name[1].name, "HessianMatrix");
                            name[1].dimSize = Idx->InputByHorizonL;
                            CHECK(cudaMemcpy(WriteHessian, Hessian, sizeof(double) * Idx->InputByHorizonL * Idx->InputByHorizonL, cudaMemcpyDeviceToHost));
                            write_Matrix_Information(WriteHessian, &name[1], timerParam);
                        }
                    }
#endif

                    /*if(t==1){
                        // CHECK_CUSOLVER( cusolverDnSgeqrf_bufferSize(cusolverH, HORIZON, HORIZON, Hessian, HORIZON, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                        CHECK_CUSOLVER( cusolverDnDgeqrf_bufferSize(cusolverH, InputByHorizon, InputByHorizon, Hessian, InputByHorizon, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                        // CHECK_CUSOLVER( cusolverDnSormqr_bufferSize(cusolverH, side, trans, HORIZON, nrhs, HORIZON, Hessian, HORIZON, hQR_tau, Gradient, HORIZON, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                        CHECK_CUSOLVER( cusolverDnDormqr_bufferSize(cusolverH, side, trans, InputByHorizon, nrhs, InputByHorizon, Hessian, InputByHorizon, hQR_tau, Gradient, InputByHorizon, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                    
                        w_si_hessian = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
                        CHECK( cudaMalloc((void**)&w_sp_hessian, sizeof(double) * w_si_hessian) );
                    }*/

                    /* compute QR factorization */ 
                    matcalc_s_t = clock();
                    CHECK_CUSOLVER( cusolverDnDgeqrf(cusolverH, Idx->InputByHorizonL, Idx->InputByHorizonL, Hessian, Idx->InputByHorizonL, hQR_tau, w_sp_hessian, w_si_hessian, devInfo),"Failed to compute QR factorization" );
                    CHECK_CUSOLVER( cusolverDnDormqr(cusolverH, side, trans, Idx->InputByHorizonL, nrhs, Idx->InputByHorizonL, Hessian, Idx->InputByHorizonL, hQR_tau, Gradient, Idx->InputByHorizonL, w_sp_hessian, w_si_hessian, devInfo), "Failed to compute Q^T*B" );
                    CHECK(cudaDeviceSynchronize());

                    // CHECK_CUBLAS( cublasStrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, HORIZON, nrhs, &m_alpha, Hessian, HORIZON, Gradient, HORIZON), "Failed to compute X = R^-1Q^T*B" );
                    CHECK_CUBLAS( cublasDtrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, Idx->InputByHorizonL, nrhs, &m_alpha, Hessian, Idx->InputByHorizonL, Gradient, Idx->InputByHorizonL), "Failed to compute X = R^-1Q^T*B" );
                    CHECK(cudaDeviceSynchronize());
                    matcalc_end_t = clock();
                    diff_time = matcalc_end_t-matcalc_s_t;
                    printf("inverseOperationForHessian := %lf\n", diff_time / CLOCKS_PER_SEC );

                    NewtonLikeMethodCopyVector<<<Idx->InputByHorizonL, 1>>>(deviceTempData, Gradient);
                    CHECK(cudaDeviceSynchronize());

                    CHECK( cudaMemcpy(hostTempData, deviceTempData, sizeof(double) * Idx->InputByHorizonL, cudaMemcpyDeviceToHost) );
                    CHECK(cudaDeviceSynchronize());
                    // NewtonLikeMethodInputSaturation(hostTempData, hostSCV->constraints[1], hostSCV->constraints[0]);
                    // Quadrotor_Input_recalculation(hostTempData, hostSCV);
                    Quadrotor_InputSaturation(hostTempData, hostSCV);
                
                    for(int uIndex = 0; uIndex < DIM_OF_INPUT; uIndex++)
                    {
                        Proposed_F[uIndex] = hostTempData[uIndex];
                    }
                    // ????????????????????????????????????->??????(vs MC???)->??????????????????????????????(by RungeKutta4.5)->???????????????
                    // calc_Cost_Quadrotor(COST_MC, hostData, hostSCV);
                    calc_Cost_QuaternionBased_Quadrotor(COST_MC, hostData, hostSCV);
                    // calc_Cost_Quadrotor(COST_NLM, hostTempData, hostSCV);
                    calc_Cost_QuaternionBased_Quadrotor(COST_NLM, hostTempData, hostSCV); 
                }               
            }
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&process_gpu_time, start, stop);
            stop_t = clock();
            procedure_all_time = stop_t - start_t;
        }
        printf("TIME stpe :: %lf", t * interval);
        printf("MCMPC cost value := %lf  Proposed cost value := %lf\n", COST_MC[0], COST_NLM[0]);
        // ??????????????????????????????????????????????????????
        if(COST_NLM[0] < COST_MC[0] /*&& optimumCondition_p[1] < optimumConditions[1]*/)
        {
            for(int j = 0; j < DIM_OF_INPUT; j++)
            {
                F_input[j] = hostTempData[j];
                // F_input[j] = hostData[j];
            }
            cost_now = COST_NLM[0];
            CHECK( cudaMemcpy(deviceData, hostTempData, sizeof(double) * Idx->InputByHorizonL, cudaMemcpyHostToDevice) );
            // CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * Idx->InputByHorizonL, cudaMemcpyHostToDevice) );
        }else{
            for(int j = 0; j < DIM_OF_INPUT; j++)
            {
                F_input[j] = hostData[j];
                // F_input[j] = hostTempData[j];
            }
            cost_now = COST_MC[0];
            CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * Idx->InputByHorizonL, cudaMemcpyHostToDevice) );
            // CHECK( cudaMemcpy(deviceData, hostTempData, sizeof(double) * InputByHorizon, cudaMemcpyHostToDevice) );
        }

        // Runge_Kutta45_for_SecondaryOderSystem( hostSCV, F_input, interval);
        // Runge_Kutta45_Quadrotor(hostSCV, F_input, interval);
        Runge_Kutta45_QuaternionBased_Quadrotor(hostSCV, F_input, interval);
        /*if(300 <= t && t <=304)
        {
            hostSCV->params[1] += 0.5;
            hostSCV->params[2] -= 0.5;
            hostSCV->params[3] += 0.5;
        }
        if(600 <= t && t <= 604){
            hostSCV->params[1] -= 0.5;
            hostSCV->params[2] += 0.5;
            hostSCV->params[3] -= 0.5;
        }*/
        if(t == 300)
        {
            hostSCV->state[1] += 1.0;
            hostSCV->state[7] += 20.0;
        }
        CHECK( cudaMemcpy(deviceSCV, hostSCV, sizeof(SystemControlVariable), cudaMemcpyHostToDevice) );
        fprintf(fp_input, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", t * interval, F_input[0], F_input[1], F_input[2], F_input[3], MCMPC_F[0], MCMPC_F[1], MCMPC_F[2], MCMPC_F[3], Proposed_F[0], Proposed_F[1], Proposed_F[2], Proposed_F[3]);
        fprintf(fp_state, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", t * interval, hostSCV->state[0], hostSCV->state[2], hostSCV->state[4], hostSCV->state[6], hostSCV->state[7], hostSCV->state[8], hostSCV->state[9], hostSCV->state[10], hostSCV->state[11], hostSCV->state[12], hostSCV->state[1], hostSCV->state[3], hostSCV->state[5]);
        fprintf(opco, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", t * interval, cost_now, COST_MC[0], COST_NLM[0], COST_MC[1], COST_NLM[1], COST_MC[1]-COST_NLM[1], process_gpu_time/10e3, procedure_all_time/CLOCKS_PER_SEC);
        printf("hoge\n");
    }
    if(cusolverH) cusolverDnDestroy(cusolverH);
    if(handle_cublas) cublasDestroy(handle_cublas);
    fclose(fp_input);
    fclose(fp_state);
    fclose(opco);
    cudaDeviceReset( );
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}