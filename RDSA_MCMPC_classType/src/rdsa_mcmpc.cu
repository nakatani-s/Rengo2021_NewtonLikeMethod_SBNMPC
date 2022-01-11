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

    time_t timeValue;
    struct tm *timeObject;
    time( &timeValue );
    timeObject = localtime( &timeValue );
    char filename1[128], filename2[128];
    sprintf(filename1,"data_input_%d%d_%d%d.txt", timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour,timeObject->tm_min);
    sprintf(filename2,"data_state_%d%d_%d%d.txt", timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour,timeObject->tm_min);
    fp_state = fopen(filename2, "w");
    fp_input = fopen(filename1, "w");

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
    info = new SampleInfo[Idx->sample_size];
    init_structure(info, Idx->sample_size, Idx->InputByHorizon);

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
    qhp = new QHP[gIdx->sample_size];
    init_structure(qhp, Idx->sample_size, Idx->HessianElements);

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

void rdsa_mcmpc::execute_rdsa_mcmpc(double *CurrentInput)
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
                                                        info, devIdx, thrust::raw_pointer_cast(sort_key_device_vec.data()));
        
        thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
        thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());
        calc_weighted_mean<<<1,1>>>(deviceDataMC, devIdx,thrust::raw_pointer_cast(indices_device_vec.data()), info);
        CHECK( cudaMemcpy(hostDataMC, deviceDataMC, sizeof(double) * gIdx->InputByHorizon, cudaMemcpyDeviceToHost) );
    }
    costValue = calc_cost(hostDataMC, _state, _parameters, _reference, _constraints, _weightMatrix, gIdx);
    printf("time step :: %lf <====> cost value :: %lf\n", time_steps * gIdx->control_cycle, costValue);
    // 予測入力を返す
    for(int i = 0; i < gIdx->dim_of_input; i++)
    {
        CurrentInput[i] = hostDataMC[i];
    }
    
    time_steps++;

}

void rdsa_mcmpc::do_forward_simulation(double *state, double *input, IntegralMethod method)
{
    double *diff_state;
    diff_state = (double *)malloc(sizeof(double) * gIdx->dim_of_state);
    switch(method)
    {
        case EULAR:
            myDynamicModel(diff_state, input, state, _parameters);
            transition_Eular(state, diff_state, gIdx->control_cycle, gIdx->dim_of_state);
            break;
        case RUNGE_KUTTA_45:
            runge_kutta_45(state, gIdx->dim_of_state, input, _parameters, gIdx->control_cycle);
            break;
        default:
            break;
    }
}

void rdsa_mcmpc::write_data_to_file(double *_input)
{
    double current_time = time_steps * gIdx->control_cycle;
    for(int i = 0; i < gIdx->dim_of_state; i++)
    {
        if(i == 0)
        {
            fprintf(fp_state,"%lf %lf ", current_time, _state[i]);
        }else if( i == gIdx->dim_of_state - 1){
            fprintf(fp_state,"%lf\n", _state[i]);
        }else{
            fprintf(fp_state,"%lf ", _state[i]);
        }
    }

    for(int i = 0; i < gIdx->dim_of_input; i++)
    {
        if(i == 0)
        {
            fprintf(fp_input, "%lf %lf ", current_time, _input[i]);
        }else if(i == gIdx->dim_of_input - 1){
            fprintf(fp_input, "%lf\n", _input[i]);
        }else{
            fprintf(fp_input, "%lf ", _input[i]);
        }
    }
}