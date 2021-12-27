/*
--------2021.12.21 start making
*/
#include "../include/rdsa_mcmpc.cuh"
// #include "../DataStructure.cuh"

// コンストラクタ
rdsa_mcmpc::rdsa_mcmpc(CoolingMethod method)
{
    time_steps = 0;
    cMethod = method;
    // インデックスパラメータの取得
    gIdx = (IndexParams*)malloc(sizeof(IndexParams));
    set_IdxParams(gIdx);
    const IndexParams *Idx = gIdx;
    CHECK( cudaMalloc(&devIdx, sizeof(IndexParams)) );
    CHECK( cudaMemcpy(devIdx, gIdx, sizeof(IndexParams), cudaMemcpyHostToDevice) );

    /* Get System & Controller Valiables */
    // hstSCV = initSCV;
    // hstSCV = (SystemControllerVariable)
    // CHECK( cudaMalloc(&devSCV, sizeof(SystemControllerVariable)) );
    // CHECK( cudaMemcpy(devSCV, hstSCV, sizeof(SystemControllerVariable), cudaMemcpyHostToDevice));

    /* Setup GPU parameters */
    randomNums = CONTROLLER::NUM_OF_SAMPLES * (OCP::DIM_OF_INPUT + 1) * CONTROLLER::HORIZON;
    numBlocks = countBlocks(CONTROLLER::NUM_OF_SAMPLES, CONTROLLER::THREAD_PER_BLOCKS);
    threadPerBlocks = CONTROLLER::THREAD_PER_BLOCKS;
    cudaMalloc((void **)&devRandSeed, randomNums * sizeof(curandState));
    setup_RandomSeed<<<CONTROLLER::NUM_OF_SAMPLES, (OCP::DIM_OF_INPUT + 1) * CONTROLLER::HORIZON>>>(devRandSeed, rand());
    cudaDeviceSynchronize();

    /* Setup Datastructure includes cost, InputSequences, ..., etc. */
    // hostSampleInfo = (SampleInfo *)malloc(sizeof(SampleInfo) * CONTROLLER::NUM_OF_SAMPLES);
    // CHECK( cudaMalloc(&devSampleInfo, sizeof(SampleInfo) * CONTROLLER::NUM_OF_SAMPLES) );
    thrust::device_vector<SampleInfo> devSampleInfo_temp(CONTROLLER::NUM_OF_SAMPLES);
    devSampleInfo = devSampleInfo_temp;

    /* 旧SystemControlVariable構造体のメンバ変数は、ユニファイドメモリで管理 */ 
    CHECK( cudaMallocManaged((void**)&_state, sizeof(double) * OCP::DIM_OF_SYSTEM_STATE) );
    CHECK( cudaMallocManaged((void**)&_reference, sizeof(double) * OCP::DIM_OF_REFERENCE) );
    CHECK( cudaMallocManaged((void**)&_parameters, sizeof(double) * OCP::DIM_OF_SYSTEM_PARAMS) );
    CHECK( cudaMallocManaged((void**)&_constraints, sizeof(double) * OCP::DIM_OF_CONSTRAINTS) );
    CHECK( cudaMallocManaged((void**)&_weightMatrix, sizeof(double) * OCP::DIM_OF_WEIGHT_MATRIX) );

    /* Matrix or Vector Array with result of Quadratic Hypersurface Regression */
    CHECK( cudaMalloc(&Hessian, sizeof(double) * Idx->HessianSize) );
    CHECK( cudaMalloc(&UpperHessian, sizeof(double) * Idx->HessianSize) );
    CHECK( cudaMalloc(&Gradient, sizeof(double) * Idx->InputByHorizon) );

    /* For executing Least-Squares method by cublas */ 
    CHECK( cudaMalloc(&CoeMatrix, sizeof(double) * Idx->PowHessianElements) );
    CHECK( cudaMalloc(&TensortX, sizeof(double) * Idx->FittingSampleSize * Idx->HessianElements) );
    CHECK( cudaMalloc(&TransposeX, sizeof(double) * Idx->FittingSampleSize * Idx->HessianElements) );
    CHECK( cudaMalloc(&TensortL, sizeof(double) * Idx->FittingSampleSize) );

    /* DataStructures for Tensor Vector per Samples */
    thrust::device_vector<QHP> devQHP_temp(CONTROLLER::NUM_OF_SAMPLES);
    devQHP = devQHP_temp;

    /* thrust ベクターの実体を定義 */
    thrust::host_vector<int> indices_host_vec_temp(CONTROLLER::NUM_OF_SAMPLES);
    indices_host_vec = indices_host_vec_temp;
    // thrust::device_vector<int> indices_device_vec_temp(CONTROLLER::NUM_OF_SAMPLES);
    // indices_device_vec = indices_device_vec_temp;
    indices_device_vec = indices_host_vec_temp;
    thrust::host_vector<double> sort_key_host_vec_temp(CONTROLLER::NUM_OF_SAMPLES);
    sort_key_host_vec = sort_key_host_vec_temp;
    sort_key_device_vec = sort_key_host_vec_temp; 

    /* コスト比較のために推定入力を一時保存するための配列の実体 */
    hostDataMC = (double *)malloc(sizeof(double) * Idx->InputByHorizon);
    hostDataRDSA =  (double *)malloc(sizeof(double) * Idx->InputByHorizon);
    CHECK( cudaMalloc(&deviceDataMC, sizeof(double) * Idx->InputByHorizon) );
    CHECK( cudaMalloc(&deviceDataRDSA, sizeof(double) * Idx->InputByHorizon) );

    // cusolver & cublas の基本設定
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH), "Failed to Create cusolver handle" );
    CHECK_CUBLAS( cublasCreate(&cublasH), "Failed to create cublas handle" );
    uplo = CUBLAS_FILL_MODE_LOWER;
    side = CUBLAS_SIDE_LEFT;
    trans = CUBLAS_OP_T;
    trans_N = CUBLAS_OP_N;
    uplo_QR = CUBLAS_FILL_MODE_UPPER;
    cub_diag = CUBLAS_DIAG_NON_UNIT;

    geqrf_work_size = 0;
    ormqr_work_size = 0;
    QR_work_size = 0;
    nrhs = 1;

}

// デストラクタ
rdsa_mcmpc::~rdsa_mcmpc()
{
    if(cusolverH) cusolverDnDestroy(cusolverH);
    if(cublasH) cublasDestroy(cublasH);
    cudaDeviceReset();
}


void rdsa_mcmpc::set(double *a, valueType type)
{
    int index = 0;
    switch(type)
    {
        case setState:
            for(int i = 0; i < OCP::DIM_OF_SYSTEM_STATE; i++)
            {
                _state[i] = a[i];
            }
            break;
        case setInput:
            for(int i = 0; i < CONTROLLER::HORIZON; i++)
            {
                for(int k = 0; k < OCP::DIM_OF_INPUT; k++)
                {
                    hostDataMC[index] = a[k];
                    index++;
                }
            }
            CHECK( cudaMemcpy(deviceDataMC, hostDataMC, sizeof(double) * gIdx->InputByHorizon, cudaMemcpyHostToDevice) );
            break;
        case setParameter:
            for(int i = 0; i < OCP::DIM_OF_SYSTEM_PARAMS; i++)
            {
                _parameters[i] = a[i];
            }
            break;
        case setConstraint:
            for(int i = 0; i < OCP::DIM_OF_CONSTRAINTS; i++)
            {
                _constraints[i] = a[i];
            }
            break;
        case setWeightMatrix:
            for(int i = 0; i < OCP::DIM_OF_WEIGHT_MATRIX; i++)
            {
                _weightMatrix[i] = a[i];
            }
            break;
        case setReference:
            for(int i = 0; i < OCP::DIM_OF_REFERENCE; i++)
            {
                _reference[i] = a[i];
            }
        default:
            break;
    }
}

void rdsa_mcmpc::execute_rdsa_mcmpc()
{
    double var;
    for(int iter = 0; iter < CONTROLLER::ITERATIONS; iter++)
    {
        switch(cMethod)
        {
            case Geometric:
                var = CONTROLLER::SIGMA * pow(CONTROLLER::c_rate, iter);
                break;
            case Hyperbolic:
                var = CONTROLLER::SIGMA / sqrt(iter + 1);
                break;
            default:
                var = CONTROLLER::SIGMA;
        }
        // parallelSimForMCMPC<<<numBlocks,threadPerBlocks>>>( var, devRandSeed, deviceDataMC, devSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data()) );
        parallelSimForMC<<<numBlocks, threadPerBlocks>>>(var, _state, _parameters, _reference, _constraints, _weightMatrix, deviceDataMC, devRandSeed, 
                                                        thrust::raw_pointer_cast(devSampleInfo.data()), devIdx, thrust::raw_pointer_cast(sort_key_device_vec.data()));


    }
    time_steps++;

}