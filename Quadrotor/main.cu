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
    /* 行列演算ライブラリ用に宣言する変数群 */
    cusolverDnHandle_t cusolverH = NULL;
    // cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH),"Failed to Create cusolver handle");

    cublasHandle_t handle_cublas = 0;
    cublasCreate(&handle_cublas);

    /* メインの実験データ書き込み用ファイルの宣言　*/
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

    /* ホスト・デバイス双方で使用するベクトルの宣言 */
    // double hostParams[DIM_OF_PARAMETERS], hostState[DIM_OF_STATES], hostConstraint[NUM_OF_CONSTRAINTS], hostWeightMatrix[DIM_OF_WEIGHT_MATRIX];
    SystemControlVariable *hostSCV, *deviceSCV;
    hostSCV = (SystemControlVariable*)malloc(sizeof(SystemControlVariable));
    init_variables( hostSCV );
    CHECK( cudaMalloc(&deviceSCV, sizeof(SystemControlVariable)) );
    CHECK( cudaMemcpy(deviceSCV, hostSCV, sizeof(SystemControlVariable), cudaMemcpyHostToDevice) );
    

    /* GPUの設定用パラメータ */
    unsigned int numBlocks, /*randomBlocks,*/ randomNums, /*Blocks,*/ dimHessian, numUnknownParamQHP, numUnknownParamHessian;
    unsigned int paramsSizeQuadHyperPlane;
    const int InputByHorizon = HORIZON * DIM_OF_INPUT;
    randomNums = NUM_OF_SAMPLES * (DIM_OF_INPUT + 1) * HORIZON;
    // randomBlocks = countBlocks(randomNums, THREAD_PER_BLOCKS);
    numBlocks = countBlocks(NUM_OF_SAMPLES, THREAD_PER_BLOCKS);
    // Blocks = numBlocks;
    dimHessian = InputByHorizon * InputByHorizon;

    numUnknownParamQHP = NUM_OF_PARABOLOID_COEFFICIENT;
    numUnknownParamHessian = numUnknownParamQHP - (InputByHorizon + 1);
    paramsSizeQuadHyperPlane = numUnknownParamQHP; //ホライズンの大きさに併せて、局所サンプルのサイズを決定
    paramsSizeQuadHyperPlane = paramsSizeQuadHyperPlane + addTermForLSM;

    dim3 block(MAX_DIVISOR,1);
    // dim3 block(1,1);
    dim3 grid((numUnknownParamQHP + block.x - 1)/ block.x, (numUnknownParamQHP + block.y -1) / block.y);
    printf("#NumBlocks = %d\n", numBlocks);
    printf("#NumBlocks = %d\n", numUnknownParamQHP);

#ifdef WRITE_MATRIX_INFORMATION
    double *WriteHessian, *WriteRegular;
    WriteHessian = (double *)malloc(sizeof(double)*dimHessian);
    WriteRegular = (double *)malloc(sizeof(double)* NUM_OF_PARABOLOID_COEFFICIENT * NUM_OF_PARABOLOID_COEFFICIENT);
    int timerParam[5] = { };
    dataName *name;
    name = (dataName*)malloc(sizeof(dataName)*5);
#endif

    /* MCMPC用の乱数生成用のseedを生成する */
    curandState *deviceRandomSeed;
    cudaMalloc((void **)&deviceRandomSeed, randomNums * sizeof(curandState));
    setup_kernel<<<NUM_OF_SAMPLES, (DIM_OF_INPUT + 1) * HORIZON>>>(deviceRandomSeed, rand());
    cudaDeviceSynchronize();
    
    /* 入力・コスト・最適性残差等の情報をまとめた構造体の宣言 */
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

    /* ２次超平面フィッティングの結果を反映する行列及びベクトルの宣言　（<---最適値計算にも使用）*/
    double *Hessian, *invHessian, *lowerHessian, *HessianElements;
    double *Gradient;
    CHECK( cudaMalloc(&Hessian, sizeof(double) * dimHessian) );
    CHECK( cudaMalloc(&invHessian, sizeof(double) * dimHessian) );
    CHECK( cudaMalloc(&lowerHessian, sizeof(double) * dimHessian) );
    CHECK( cudaMalloc(&HessianElements, sizeof(double) * numUnknownParamQHP) );

    CHECK( cudaMalloc(&Gradient, sizeof(double) * InputByHorizon) );

    /* 最小２乗法で２次超曲面を求める際に使用する配列の宣言 */
    double *Gmatrix, *invGmatrix, *CVector, *ansCVector;
    CHECK( cudaMalloc(&CVector, sizeof(double) * numUnknownParamQHP) );
    CHECK( cudaMalloc(&ansCVector, sizeof(double) * numUnknownParamQHP) );
    CHECK( cudaMalloc(&Gmatrix, sizeof(double) * numUnknownParamQHP * numUnknownParamQHP) );
    CHECK( cudaMalloc(&invGmatrix, sizeof(double) * numUnknownParamQHP * numUnknownParamQHP) );


    QHP *deviceQHP;
    CHECK( cudaMalloc(&deviceQHP, sizeof(QHP) * paramsSizeQuadHyperPlane) );

    unsigned int qhpBlocks;
    qhpBlocks = countBlocks(paramsSizeQuadHyperPlane, THREAD_PER_BLOCKS);
    printf("#qhpBlocks = %d\n", qhpBlocks);

    // 行列演算ライブラリ用の変数の宣言及び定義
    const int m_Rmatrix = numUnknownParamQHP;

    int work_size, w_si_hessian;
    double *work_space, *w_sp_hessian;
    int *devInfo;
    CHECK( cudaMalloc((void**)&devInfo, sizeof(int) ) );

    /* thrust使用のためのホスト/デバイス用ベクトルの宣言 */ 
    thrust::host_vector<int> indices_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<int> indices_device_vec = indices_host_vec;
    thrust::host_vector<double> sort_key_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<double> sort_key_device_vec = sort_key_host_vec; 

    /* 推定入力のプロット・データ転送用　*/
    double *hostData, *deviceData, *hostTempData, *deviceTempData;
    hostData = (double *)malloc(sizeof(double) * InputByHorizon);
    hostTempData = (double *)malloc(sizeof(double) * InputByHorizon);
    CHECK(cudaMalloc(&deviceData, sizeof(double) * InputByHorizon));
    CHECK(cudaMalloc(&deviceTempData, sizeof(double) * InputByHorizon));
    for(int i = 0; i < (HORIZON * DIM_OF_INPUT) ; i++){
        if(i % 4 == 0){
            hostData[i] = hostSCV->params[0];
        }
    }
    hostData[0] = hostSCV->params[0];
    CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * InputByHorizon, cudaMemcpyHostToDevice));

    /* 制御ループの開始 */
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
    clock_t start_t, stop_t;
    cudaEvent_t start, stop;
    
    dim3 inverseGmatrix(numUnknownParamQHP, numUnknownParamQHP);
    // dim3 grid_inverse(HORIZON, HORIZON);
    dim3 grid_inverse(InputByHorizon, InputByHorizon);
    dim3 threads((HORIZON + grid_inverse.x -1) / grid_inverse.x, (HORIZON + grid_inverse.y -1) / grid_inverse.y);

#ifdef USING_QR_DECOMPOSITION
    // double *QR_work_space = NULL;
    double *ws_QR_operation = NULL;
    int geqrf_work_size = 0;
    int ormqr_work_size = 0;
    int QR_work_size = 0;
    const int nrhs = 1;
    double *QR_tau = NULL;
    double *hQR_tau = NULL;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasOperation_t trans_N = CUBLAS_OP_N;
    cublasFillMode_t uplo_QR = CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t cub_diag = CUBLAS_DIAG_NON_UNIT;
    CHECK(cudaMalloc((void**)&QR_tau, sizeof(double) * numUnknownParamQHP));
    CHECK(cudaMalloc((void**)&hQR_tau, sizeof(double) * InputByHorizon));
#endif

    mInputSystem tsDims = MultiInput;
    for(int t = 0; t < SIM_TIME; t++)
    {
        shift_Input_vec( hostData );
        CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * InputByHorizon, cudaMemcpyHostToDevice) );
        start_t = clock();

        if(t == 0)
        {
            start_t = clock();
            for(int iter = 0; iter < ITERATIONS_MAX; iter++)
            {
                var = variance / sqrt(iter + 1);
                // var = variance / 2;
                MCMPC_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                // MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                cudaDeviceSynchronize();
                thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());

                
                getEliteSampleInfo_multiInput<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                // getEliteSampleInfo<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                CHECK( cudaMemcpy(hostEliteSampleInfo, deviceEliteSampleInfo, sizeof(SampleInfo) * NUM_OF_ELITES, cudaMemcpyDeviceToHost) );
                // weighted_mean(hostData, NUM_OF_ELITES, hostSampleInfo);
                weighted_mean_multiInput(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
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
                CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * InputByHorizon, cudaMemcpyHostToDevice) );
                calc_Cost_Quadrotor(COST_MC, hostData, hostSCV);
                // calc_OC_for_Cart_and_SinglePole_hostF(optimumConditions, hostData, hostSCV, hostTol);
                // printf("cost :: %lf   KKT_Error :: %lf\n", optimumConditions[0], optimumConditions[1]);
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
                var = variance / 2.0;
                MCMPC_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                // MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                cudaDeviceSynchronize();
                thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());
            
                // CHECK( cudaMemcpy(hostSampleInfo, deviceSampleInfo, sizeof(SampleInfo) * NUM_OF_SAMPLES, cudaMemcpyDeviceToHost) );
                getEliteSampleInfo_multiInput<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                // getEliteSampleInfo<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                CHECK( cudaMemcpy(hostEliteSampleInfo, deviceEliteSampleInfo, sizeof(SampleInfo) * NUM_OF_ELITES, cudaMemcpyDeviceToHost) );
                weighted_mean_multiInput(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
                // weighted_mean(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
                for(int uIndex = 0; uIndex < DIM_OF_INPUT; uIndex++)
                {
                    MCMPC_F[uIndex] = hostData[uIndex];
                }
                
                CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * HORIZON, cudaMemcpyHostToDevice) );
                var = neighborVar;
                MCMPC_Quadrotor<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                // MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                cudaDeviceSynchronize();
                thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());

                NewtonLikeMethodGetTensorVectorTest<<< qhpBlocks, THREAD_PER_BLOCKS>>>(deviceQHP, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ),tsDims);
                // NewtonLikeMethodGetTensorVector<<< qhpBlocks, THREAD_PER_BLOCKS>>>(deviceQHP, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                // NewtonLikeMethodGetTensorVectorNormarizationed<<< qhpBlocks, THREAD_PER_BLOCKS>>>(deviceQHP, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ), deviceSCV);
                cudaDeviceSynchronize();

                // 1024以下の"NUM_OF_PARABOLOID_COEFFICIENT"の最大約数を(thread数 / block)として計算させる方針で実行
                // 以下は正規方程式における行列の各要素を取得する関数
                if(NUM_OF_PARABOLOID_COEFFICIENT < 1024){
                    NewtonLikeMethodGetRegularMatrix<<<NUM_OF_PARABOLOID_COEFFICIENT, NUM_OF_PARABOLOID_COEFFICIENT>>>(Gmatrix, deviceQHP, paramsSizeQuadHyperPlane);
                }else{
                    NewtonLikeMethodGetRegularMatrixTypeB<<<grid, block>>>(Gmatrix, deviceQHP, paramsSizeQuadHyperPlane, NUM_OF_PARABOLOID_COEFFICIENT);
                }
                // NewtonLikeMethodGenNormalizationMatrix<<<grid, block>>>(Gmatrix, deviceQHP, paramsSizeQuadHyperPlane, NUM_OF_PARABOLOID_COEFFICIENT);

                /*-----------------Error detect 2021.07.20----------------------------*/
                // Following Function has any Error (ThreadId or BlockId) --> it is required to modify original mode.
                // NewtonLikeMethodGenNormalEquation<<<grid, block>>>(Gmatrix, CVector, deviceQHP, paramsSizeQuadHyperPlane, NUM_OF_PARABOLOID_COEFFICIENT);
                // NewtonLikeMethodGetRegularMatrix<<<NUM_OF_PARABOLOID_COEFFICIENT, NUM_OF_PARABOLOID_COEFFICIENT>>>(Gmatrix, deviceQHP, paramsSizeQuadHyperPlane);
                NewtonLikeMethodGetRegularVector<<<NUM_OF_PARABOLOID_COEFFICIENT, 1>>>(CVector, deviceQHP, paramsSizeQuadHyperPlane);
                // printf("hoge %d hoge\n",t);
                cudaDeviceSynchronize();
#ifdef WRITE_MATRIX_INFORMATION
                if(t<100){
                    if(t % 10 == 0){
                        get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t);
                        sprintf(name[0].name, "RegularMatrix");
                        name[0].dimSize = NUM_OF_PARABOLOID_COEFFICIENT;
                        CHECK(cudaMemcpy(WriteRegular, Gmatrix, sizeof(double) * NUM_OF_PARABOLOID_COEFFICIENT * NUM_OF_PARABOLOID_COEFFICIENT, cudaMemcpyDeviceToHost));
                        write_Matrix_Information(WriteRegular, &name[0], timerParam);
                    }
                }else{
                    if(t % 250 == 0){
                        get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t);
                        sprintf(name[0].name, "RegularMatrix");
                        name[0].dimSize = NUM_OF_PARABOLOID_COEFFICIENT;
                        CHECK(cudaMemcpy(WriteRegular, Gmatrix, sizeof(double) * NUM_OF_PARABOLOID_COEFFICIENT * NUM_OF_PARABOLOID_COEFFICIENT, cudaMemcpyDeviceToHost));
                        write_Matrix_Information(WriteRegular, &name[0], timerParam);
                    }

                }
#endif

#ifndef USING_QR_DECOMPOSITION
                //以下は、正規方程式（最小二乗法で使用）のベクトル(正規方程式：Gx = v の v)の各要素を計算する関数
                // NewtonLikeMethodGenNormalizationVector<<<NUM_OF_PARABOLOID_COEFFICIENT, 1>>>(CVector, deviceQHP, paramsSizeQuadHyperPlane);
                // cudaDeviceSynchronize();

                // CHECK_CUSOLVER( cusolverDnSpotrf_bufferSize(cusolverH, uplo, m_Rmatrix, Gmatrix, m_Rmatrix, &work_size), "Failed to get bufferSize");
                CHECK_CUSOLVER( cusolverDnDpotrf_bufferSize(cusolverH, uplo, m_Rmatrix, Gmatrix, m_Rmatrix, &work_size), "Failed to get bufferSize");
                CHECK(cudaMalloc((void**)&work_space, sizeof(double) * work_size));

                // CHECK_CUSOLVER( cusolverDnSpotrf(cusolverH, uplo, m_Rmatrix, Gmatrix, m_Rmatrix, work_space, work_size, devInfo), "Failed to inverse operation for G");
                CHECK_CUSOLVER( cusolverDnDpotrf(cusolverH, uplo, m_Rmatrix, Gmatrix, m_Rmatrix, work_space, work_size, devInfo), "Failed to inverse operation for G");
                MatrixSetUpLargeIdentityMatrix<<<grid, block>>>(invGmatrix, NUM_OF_PARABOLOID_COEFFICIENT);
                cudaDeviceSynchronize();

                // CHECK_CUSOLVER( cusolverDnSpotrs(cusolverH, uplo, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, invGmatrix, m_Rmatrix, devInfo), "Failed to get inverse Matrix G");
                CHECK_CUSOLVER( cusolverDnDpotrs(cusolverH, uplo, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, invGmatrix, m_Rmatrix, devInfo), "Failed to get inverse Matrix G");

                // 正規方程式をcuBlasで解く
                // CHECK_CUBLAS( cublasSgemv(handle_cublas, CUBLAS_OP_N, m_Rmatrix, m_Rmatrix, &alpha, invGmatrix, m_Rmatrix, CVector, 1, &beta, ansCVector, 1),"Failed to get Estimate Input Sequences");
                CHECK_CUBLAS( cublasDgemv(handle_cublas, CUBLAS_OP_N, m_Rmatrix, m_Rmatrix, &alpha, invGmatrix, m_Rmatrix, CVector, 1, &beta, ansCVector, 1),"Failed to get Estimate Input Sequences");

#else
                if(t==1){
                    // CHECK_CUSOLVER( cusolverDnSgeqrf_bufferSize(cusolverH, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                    CHECK_CUSOLVER( cusolverDnDgeqrf_bufferSize(cusolverH, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                    // CHECK_CUSOLVER( cusolverDnSormqr_bufferSize(cusolverH, side, trans, m_Rmatrix, nrhs, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, CVector, m_Rmatrix, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                    CHECK_CUSOLVER( cusolverDnDormqr_bufferSize(cusolverH, side, trans, m_Rmatrix, nrhs, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, CVector, m_Rmatrix, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );

                    QR_work_size = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
                }
                CHECK( cudaMalloc((void**)&ws_QR_operation, sizeof(double) * QR_work_size) );
                /* compute QR factorization */ 
                // CHECK_CUSOLVER( cusolverDnSgeqrf(cusolverH, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, ws_QR_operation, QR_work_size, devInfo),"Failed to compute QR factorization" );
                CHECK_CUSOLVER( cusolverDnDgeqrf(cusolverH, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, ws_QR_operation, QR_work_size, devInfo),"Failed to compute QR factorization" );

                // CHECK_CUSOLVER( cusolverDnSormqr(cusolverH, side, trans, m_Rmatrix, nrhs, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, CVector, m_Rmatrix, ws_QR_operation, QR_work_size, devInfo), "Failed to compute Q^T*B" );
                CHECK_CUSOLVER( cusolverDnDormqr(cusolverH, side, trans, m_Rmatrix, nrhs, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, CVector, m_Rmatrix, ws_QR_operation, QR_work_size, devInfo), "Failed to compute Q^T*B" );
                CHECK(cudaDeviceSynchronize());

                // CHECK_CUBLAS( cublasStrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, m_Rmatrix, nrhs, &alpha, Gmatrix, m_Rmatrix, CVector, m_Rmatrix), "Failed to compute X = R^-1Q^T*B" );
                CHECK_CUBLAS( cublasDtrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, m_Rmatrix, nrhs, &alpha, Gmatrix, m_Rmatrix, CVector, m_Rmatrix), "Failed to compute X = R^-1Q^T*B" );

                CHECK(cudaDeviceSynchronize());

                NewtonLikeMethodCopyVector<<<numUnknownParamQHP, 1>>>(ansCVector, CVector);
                CHECK(cudaDeviceSynchronize());
#endif

                NewtonLikeMethodGetHessianElements<<<numUnknownParamHessian, 1>>>(HessianElements, ansCVector);
                CHECK(cudaDeviceSynchronize());
                // ヘシアンの上三角行列分の要素を取得
                NewtonLikeMethodGetHessianOriginal<<<InputByHorizon, InputByHorizon>>>(Hessian, HessianElements);
                CHECK(cudaDeviceSynchronize());

                NewtonLikeMethodGetLowerTriangle<<<InputByHorizon, InputByHorizon>>>(lowerHessian, Hessian);
                CHECK(cudaDeviceSynchronize());
                // NewtonLikeMethodGetFullHessianLtoU<<<HORIZON, HORIZON>>>(Hessian, lowerHessian);
                NewtonLikeMethodGetFullHessianUtoL<<<InputByHorizon, InputByHorizon>>>(lowerHessian, Hessian);
                NewtonLikeMethodGetGradient<<<InputByHorizon, 1>>>(Gradient, ansCVector, numUnknownParamHessian);
                MatrixMultiplyOperation<<<InputByHorizon,InputByHorizon>>>(Hessian, 2.0, lowerHessian);

#ifdef WRITE_MATRIX_INFORMATION
                if(t<10){
                    if(t % 1 == 0){
                        get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t);
                        sprintf(name[1].name, "HessianMatrix");
                        name[1].dimSize = HORIZON;
                        CHECK(cudaMemcpy(WriteHessian, Hessian, sizeof(double) * HORIZON * HORIZON, cudaMemcpyDeviceToHost));
                        write_Matrix_Information(WriteHessian, &name[1], timerParam);
                    }
                }
#endif

#ifndef USING_QR_DECOMPOSITION
                // CHECK_CUSOLVER( cusolverDnSpotrf_bufferSize(cusolverH, uplo, HORIZON, Hessian, HORIZON, &w_si_hessian), "Failed to get bufferSize of computing the inverse of Hessian");
                CHECK_CUSOLVER( cusolverDnDpotrf_bufferSize(cusolverH, uplo,InputByHorizon, Hessian, InputByHorizon, &w_si_hessian), "Failed to get bufferSize of computing the inverse of Hessian");
                CHECK( cudaMalloc((void**)&w_sp_hessian, sizeof(double) * w_si_hessian) );
                // CHECK_CUSOLVER( cusolverDnSpotrf(cusolverH, uplo, HORIZON, Hessian, HORIZON, w_sp_hessian, w_si_hessian, devInfo), "Failed to inverse operation");
                CHECK_CUSOLVER( cusolverDnDpotrf(cusolverH, uplo, InputByHorizon, Hessian, InputByHorizon, w_sp_hessian, w_si_hessian, devInfo), "Failed to inverse operation");
                MatrixSetUpSmallIdentityMatrix<<<InputByHorizon, InputByHorizon>>>(invHessian);

                // CHECK_CUSOLVER( cusolverDnSpotrs(cusolverH, uplo, HORIZON, HORIZON, Hessian, HORIZON, invHessian, HORIZON, devInfo), "Failed to get inverse of Hessian");
                CHECK_CUSOLVER( cusolverDnDpotrs(cusolverH, uplo, InputByHorizon, InputByHorizon, Hessian, InputByHorizon, invHessian, InputByHorizon, devInfo), "Failed to get inverse of Hessian");
                // 逆行列を-1倍する操作
                MatrixMultiplyOperation<<<InputByHorizon, InputByHorizon>>>(Hessian, -1.0f, invHessian);
                // CHECK_CUBLAS(cublasSgemv(handle_cublas, CUBLAS_OP_N, HORIZON, HORIZON, &alpha, Hessian, HORIZON, Gradient, 1, &beta, deviceTempData, 1), "Failed to get result by proposed method");
                CHECK_CUBLAS(cublasDgemv(handle_cublas, CUBLAS_OP_N, InputByHorizon, InputByHorizon, &alpha, Hessian, InputByHorizon, Gradient, 1, &beta, deviceTempData, 1), "Failed to get result by proposed method");
#else
                if(t==1){
                    // CHECK_CUSOLVER( cusolverDnSgeqrf_bufferSize(cusolverH, HORIZON, HORIZON, Hessian, HORIZON, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                    CHECK_CUSOLVER( cusolverDnDgeqrf_bufferSize(cusolverH, InputByHorizon, InputByHorizon, Hessian, InputByHorizon, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                    // CHECK_CUSOLVER( cusolverDnSormqr_bufferSize(cusolverH, side, trans, HORIZON, nrhs, HORIZON, Hessian, HORIZON, hQR_tau, Gradient, HORIZON, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                    CHECK_CUSOLVER( cusolverDnDormqr_bufferSize(cusolverH, side, trans, InputByHorizon, nrhs, InputByHorizon, Hessian, InputByHorizon, hQR_tau, Gradient, InputByHorizon, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                    
                    w_si_hessian = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
                }
                CHECK( cudaMalloc((void**)&w_sp_hessian, sizeof(double) * w_si_hessian) );
                /* compute QR factorization */ 

                // CHECK_CUSOLVER( cusolverDnSgeqrf(cusolverH, HORIZON, HORIZON, Hessian, HORIZON, hQR_tau, w_sp_hessian, w_si_hessian, devInfo),"Failed to compute QR factorization" );
                CHECK_CUSOLVER( cusolverDnDgeqrf(cusolverH, InputByHorizon, InputByHorizon, Hessian, InputByHorizon, hQR_tau, w_sp_hessian, w_si_hessian, devInfo),"Failed to compute QR factorization" );
                // CHECK_CUSOLVER( cusolverDnSormqr(cusolverH, side, trans, HORIZON, nrhs, HORIZON, Hessian, HORIZON, hQR_tau, Gradient, HORIZON, w_sp_hessian, w_si_hessian, devInfo), "Failed to compute Q^T*B" );
                CHECK_CUSOLVER( cusolverDnDormqr(cusolverH, side, trans, InputByHorizon, nrhs, InputByHorizon, Hessian, InputByHorizon, hQR_tau, Gradient, InputByHorizon, w_sp_hessian, w_si_hessian, devInfo), "Failed to compute Q^T*B" );
                CHECK(cudaDeviceSynchronize());

                // CHECK_CUBLAS( cublasStrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, HORIZON, nrhs, &m_alpha, Hessian, HORIZON, Gradient, HORIZON), "Failed to compute X = R^-1Q^T*B" );
                CHECK_CUBLAS( cublasDtrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, InputByHorizon, nrhs, &m_alpha, Hessian, InputByHorizon, Gradient, InputByHorizon), "Failed to compute X = R^-1Q^T*B" );
                CHECK(cudaDeviceSynchronize());

                NewtonLikeMethodCopyVector<<<InputByHorizon, 1>>>(deviceTempData, Gradient);
                CHECK(cudaDeviceSynchronize());
#endif
                CHECK( cudaMemcpy(hostTempData, deviceTempData, sizeof(double) * InputByHorizon, cudaMemcpyDeviceToHost) );
                CHECK(cudaDeviceSynchronize());
                // NewtonLikeMethodInputSaturation(hostTempData, hostSCV->constraints[1], hostSCV->constraints[0]);
                Quadrotor_InputSaturation(hostTempData, hostSCV);
                for(int uIndex = 0; uIndex < DIM_OF_INPUT; uIndex++)
                {
                    Proposed_F[uIndex] = hostTempData[uIndex];
                }
                // 提案法の最適性条件を計算->比較(vs MC解)->物理シミュレーション(by RungeKutta4.5)->結果の保存
                calc_Cost_Quadrotor(COST_MC, hostData, hostSCV);
                calc_Cost_Quadrotor(COST_NLM, hostTempData, hostSCV);
                
                
            }
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&process_gpu_time, start, stop);
            stop_t = clock();
            procedure_all_time = stop_t - start_t;
        }
        printf("TIME stpe :: %lf", t * interval);
        printf("MCMPC cost value := %lf  Proposed cost value := %lf\n", COST_MC[0], COST_NLM[0]);
        // 評価値比較に基づく投入する入力の決定
        if(COST_NLM[0] < COST_MC[0] /*&& optimumCondition_p[1] < optimumConditions[1]*/)
        {
            for(int j = 0; j < DIM_OF_INPUT; j++)
            {
                F_input[j] = hostTempData[j];
            }
            cost_now = COST_NLM[0];
            CHECK( cudaMemcpy(deviceData, hostTempData, sizeof(double) * InputByHorizon, cudaMemcpyHostToDevice) );
        }else{
            for(int j = 0; j < DIM_OF_INPUT; j++)
            {
                F_input[j] = hostData[j];
            }
            cost_now = COST_MC[0];
            CHECK( cudaMemcpy(deviceData, hostData, sizeof(double) * InputByHorizon, cudaMemcpyHostToDevice) );
        }

        // Runge_Kutta45_for_SecondaryOderSystem( hostSCV, F_input, interval);
        Runge_Kutta45_Quadrotor(hostSCV, F_input, interval);
        CHECK( cudaMemcpy(deviceSCV, hostSCV, sizeof(SystemControlVariable), cudaMemcpyHostToDevice) );
        fprintf(fp_input, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", t * interval, F_input[0], F_input[1], F_input[2], F_input[3], MCMPC_F[0], MCMPC_F[1], MCMPC_F[2], MCMPC_F[3], Proposed_F[0], Proposed_F[1], Proposed_F[2], Proposed_F[3]);
        fprintf(fp_state, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", t * interval, hostSCV->state[0], hostSCV->state[2], hostSCV->state[4], hostSCV->state[6], hostSCV->state[7], hostSCV->state[8], hostSCV->state[1], hostSCV->state[3], hostSCV->state[5]);
        fprintf(opco, "%lf %lf %lf %lf %lf %lf %lf %lf\n", t * interval, cost_now, COST_MC[0], COST_NLM[0], COST_MC[1], COST_NLM[1], process_gpu_time/10e3, procedure_all_time/CLOCKS_PER_SEC);
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