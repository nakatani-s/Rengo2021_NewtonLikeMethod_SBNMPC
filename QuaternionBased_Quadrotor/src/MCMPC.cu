/*
    MCMPC.cu
*/
#include<stdio.h>
#include "../include/MCMPC.cuh"

void shift_Input_vec( double *inputVector)
{
    double temp[HORIZON]= { };
    for(int i = 0; i < HORIZON - 1; i++){
        temp[i] = inputVector[i+1];
    }
    temp[HORIZON - 1] = inputVector[HORIZON - 1];
    for(int i = 0; i < HORIZON; i++){
        inputVector[i] = temp[i];
    }
}


/*void weighted_mean(double *Output, int num_elite, SampleInfo *hInfo)
{
    double totalWeight = 0.0;
    double temp[HORIZON] = { };
    for(int i = 0; i < num_elite; i++){
        if(isnan(hInfo[i].W)){
            totalWeight += 0.0;
        }else{
            totalWeight += hInfo[i].W;
        }
    }
    for(int i = 0; i < HORIZON; i++){
        for(int k = 0; k < num_elite; k++){
            if(isnan(hInfo[k].W))
            {
                temp[i] += 0.0;
            }else{
                temp[i] += (hInfo[k].W * hInfo[k].Input[i]) / totalWeight;
            }
        }
        if(isnan(temp[i]))
        {
            Output[i] = 0.0;
        }else{
            Output[i] = temp[i];
        }
    }
}*/

void weighted_mean_multiInput(double *Output, int num_elite, SampleInfo *hInfo)
{
    double totalWeight = 0.0;
    double temp[HORIZON * DIM_OF_INPUT] = { };
    for(int i = 0; i < num_elite; i++){
        if(isnan(hInfo[i].W)){
            totalWeight += 0.0;
        }else{
            totalWeight += hInfo[i].W;
        }
    }
    int U_ID;
    for(int i = 0; i < HORIZON; i++){
        for(int uIndex = 0; uIndex < DIM_OF_INPUT; uIndex++){
            U_ID = i * DIM_OF_INPUT + uIndex;
            for(int k = 0; k < num_elite; k++){
                if(isnan(hInfo[k].W))
                {
                    temp[U_ID] += 0.0;
                }else{
                    temp[U_ID] += (hInfo[k].W * hInfo[k].Input[uIndex][i]) / totalWeight;
                }
            }
            if(isnan(temp[U_ID]))
            {
                Output[U_ID] = 0.0;
            }else{
                Output[U_ID] = temp[U_ID];
            }
        }
    }
}

void weighted_mean_multiInput_Quadrotor(double *Output, int num_elite, SampleInfo *hInfo, SystemControlVariable *SCV)
{
    double totalWeight = 0.0;
    double temp[HORIZON * DIM_OF_INPUT] = { };
    for(int i = 0; i < num_elite; i++){
        if(isnan(hInfo[i].W)){
            totalWeight += 0.0;
        }else{
            totalWeight += hInfo[i].W;
        }
    }
    int U_ID;
    for(int i = 0; i < HORIZON; i++){
        for(int uIndex = 0; uIndex < DIM_OF_INPUT; uIndex++){
            U_ID = i * DIM_OF_INPUT + uIndex;
            for(int k = 0; k < num_elite; k++){
                if(isnan(hInfo[k].W))
                {
                    temp[U_ID] += 0.0;
                }else{
                    temp[U_ID] += (hInfo[k].W * hInfo[k].Input[uIndex][i]) / totalWeight;
                }
            }
            if(isnan(temp[U_ID]))
            {
                Output[U_ID] = 0.0;
            }else{
                if(U_ID % 4 == 0){
                    Output[U_ID] = temp[U_ID] * SCV->constraints[5];
                }else{
                    Output[U_ID] = temp[U_ID];
                }
            }
        }
    }
}



__global__ void setup_kernel(curandState *state,int seed)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, id, 0, &state[id]);
}

/*__global__ void getEliteSampleInfo( SampleInfo *Elite, SampleInfo *All, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Elite[id].W = All[indices[id]].W;
    Elite[id].L = All[indices[id]].L;
    for(int i = 0; i < HORIZON; i++)
    {
        Elite[id].Input[i] = All[indices[id]].Input[i];
    }
    __syncthreads();
}*/

__global__ void getEliteSampleInfo_multiInput(SampleInfo *Elite, SampleInfo *All, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Elite[id].W = All[indices[id]].W;
    Elite[id].L = All[indices[id]].L;
    for(int i = 0; i < HORIZON; i++)
    {
        for(int uIndex = 0; uIndex < DIM_OF_INPUT; uIndex++){
            Elite[id].Input[uIndex][i] = All[indices[id]].Input[uIndex][i];
        }
    }
    __syncthreads( );
}

__device__ double gen_u(unsigned int id, curandState *state, double ave, double vr) {
    double u;
    curandState localState = state[id];
    u = curand_normal(&localState) * vr + ave;
    return u;
}

__device__ void gen_multi_input(double *u, unsigned int id, curandState *state, const int step, double *ave, double var)
{
    int local_ID = id;
    int input_ID;
    curandState localState;
    for(int i = 0; i < DIM_OF_INPUT; i++)
    {
        localState = state[local_ID];
        input_ID = (DIM_OF_INPUT * step) + i;
        u[input_ID] = curand_normal(&localState) * var + ave[i];
        local_ID += (HORIZON * DIM_OF_INPUT);
    }
}



/*__global__ void MCMPC_Cart_and_SinglePole( SystemControlVariable *SCV, double var, curandState *randomSeed, double *mean, SampleInfo *Info, double *cost_vec)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq = id;

    double stageCost = 0.0;
    double totalCost = 0.0;
    double logBarrier = 0.0;
    double u[HORIZON] = { };
    double stateInThisThreads[DIM_OF_STATES] = { };
    double dstateInThisThreads[DIM_OF_STATES] = { };

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        stateInThisThreads[i] = SCV->state[i];
    }

    double d_sec = predictionInterval / HORIZON;
    for(int t = 0; t < HORIZON; t++)
    {
        if(isnan(mean[t])){
            if(t < HORIZON -1){
                u[t] = gen_u(seq, randomSeed, Info[0].Input[t+1], var);
                seq += HORIZON;
            }else{
                u[t] = gen_u(seq, randomSeed, Info[0].Input[HORIZON - 1], var);
                seq += HORIZON;
            }
        }else{
            u[t] = gen_u(seq, randomSeed, mean[t], var);
            seq += HORIZON;
        }
#ifdef InputSaturation
        if(u[t] < SCV->constraints[0]){
            u[t] = SCV->constraints[0] + zeta;
        }
        if(u[t] > SCV->constraints[1]){
            u[t] = SCV->constraints[1] - zeta;
        }
#endif
        //まずは、(Δt = prediction interval(s) / HORIZON (step))の刻み幅でオイラー積分する方法
        dstateInThisThreads[0] = stateInThisThreads[2]; // dx_{cur}
        dstateInThisThreads[1] = stateInThisThreads[3]; // dTheta_{cur}
        dstateInThisThreads[2] = Cart_type_Pendulum_ddx(u[t], stateInThisThreads[0], stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], SCV);
        dstateInThisThreads[3] = Cart_type_Pendulum_ddtheta(u[t], stateInThisThreads[0],  stateInThisThreads[1], stateInThisThreads[2], stateInThisThreads[3], SCV);
        stateInThisThreads[2] = stateInThisThreads[2] + (d_sec * dstateInThisThreads[2]);
        stateInThisThreads[3] = stateInThisThreads[3] + (d_sec * dstateInThisThreads[3]);
        stateInThisThreads[0] = stateInThisThreads[0] + (d_sec * dstateInThisThreads[0]);
        stateInThisThreads[1] = stateInThisThreads[1] + (d_sec * dstateInThisThreads[1]);

        logBarrier = -logf(u[t] + SCV->constraints[1]) - logf(SCV->constraints[1] - u[t]) + (SCV->constraints[1] - SCV->constraints[0]) * sRho;
        stageCost = stateInThisThreads[0] * stateInThisThreads[0] * SCV->weightMatrix[0] + sinf(stateInThisThreads[1] / 2) * sinf(stateInThisThreads[1] / 2) * SCV->weightMatrix[1]
                    + stateInThisThreads[2] * stateInThisThreads[2] * SCV->weightMatrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * SCV->weightMatrix[3]
                    + u[t] * u[t] * SCV->weightMatrix[4];
        
        if(t == HORIZON -1){
            stageCost += stateInThisThreads[0] * stateInThisThreads[0] * SCV->weightMatrix[0] + sinf(stateInThisThreads[1] / 2) * sinf(stateInThisThreads[1] / 2) * SCV->weightMatrix[1]
                        + stateInThisThreads[2] * stateInThisThreads[2] * SCV->weightMatrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * SCV->weightMatrix[3];
        }

        totalCost += stageCost + Rho * logBarrier;
        logBarrier = 0.0;
        stageCost = 0.0;
    }
    double KL_COST, S, lambda, HM_COST, HM;
    lambda = mic * HORIZON;
    HM = totalCost / (hdelta * HORIZON);
    S = totalCost / lambda;
    KL_COST = exp(-S);
    HM_COST = exp(-HM);
    __syncthreads();

    Info[id].W = KL_COST;
    Info[id].L = totalCost / HORIZON;
    Info[id].WHM = HM_COST;
    cost_vec[id] = totalCost;
    for(int index = 0; index < HORIZON; index++){
        Info[id].Input[index] = u[index];
    }
    __syncthreads();
}*/

__global__ void MCMPC_Quadrotor( SystemControlVariable *SCV, double var, curandState *randomSeed, double *mean, SampleInfo *Info, double *cost_vec)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq = id;
    int init_ID;
    double stageCost = 0.0;
    double totalCost = 0.0;
    double totalCost_Fit = 0.0;
    double logBarrier = 0.0;
    double u[HORIZON * DIM_OF_INPUT] = { };
    double current_guess[DIM_OF_INPUT] = { };
    double stateInThisThreads[DIM_OF_STATES] = { };
    double dstateInThisThreads[DIM_OF_STATES] = { };

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        stateInThisThreads[i] = SCV->state[i];
    }

    double d_sec = predictionInterval / HORIZON;
    for(int t = 0; t < HORIZON; t++)
    {
        init_ID = t * DIM_OF_INPUT;
        int index_input;
        // gen_multi_input(seq, randomSeed, )
        if(isnan(mean[t*DIM_OF_INPUT]) || isnan(mean[t*DIM_OF_INPUT+1]) || isnan(mean[t*DIM_OF_INPUT+2]) || isnan(mean[t*DIM_OF_INPUT+3])){
            for(int id_cg = 0; id_cg < DIM_OF_INPUT; id_cg++)
            {
                // current_guess[id_cg] = Info[0].Input[id_cg][t];
                index_input = t * DIM_OF_INPUT + id_cg;
                current_guess[id_cg] = mean[index_input];
            }
        }else{
            for(int id_cg = 0; id_cg < DIM_OF_INPUT; id_cg++)
            {
                index_input = t * DIM_OF_INPUT + id_cg;
                current_guess[id_cg] = mean[index_input];
            }
        }
        gen_multi_input(u, seq, randomSeed, t, current_guess, var);
        seq += NUM_OF_SAMPLES;
        /*if(id % 5000 == 0){
            printf("id::= %d, u[0]=:%lf, u[1] = %lf, u[2] = %lf, u[3] = %lf\n", id,  mean[init_ID], u[init_ID+1], u[init_ID+2], u[init_ID+3]);
        }*/
#ifdef InputSaturation
        if(u[init_ID] < SCV->constraints[4])
        {
            u[init_ID] = SCV->constraints[4] + zeta;
        }
        if(u[init_ID] > SCV->constraints[5])
        {
            u[init_ID] = SCV->constraints[5] - zeta;
        }
        for(int c_i = 1; c_i < DIM_OF_INPUT; c_i++)
        {
            if(u[init_ID+c_i] < SCV->constraints[2]){
                u[init_ID+c_i] = SCV->constraints[2] + zeta;
            }
            if(u[init_ID+c_i] > SCV->constraints[3]){
                u[init_ID+c_i] = SCV->constraints[3] - zeta;
            }
        }
#endif
        //まずは、(Δt = prediction interval(s) / HORIZON (step))の刻み幅でオイラー積分する方法
        dynamics_ddot_Quadrotor(dstateInThisThreads, u[init_ID], u[init_ID + 1], u[init_ID + 2], u[init_ID + 3], stateInThisThreads, SCV); // get dx, ddx, dy, ddy, dz, ddz, dγ, dβ, dα
        for(int st_id = 0; st_id < DIM_OF_STATES; st_id++)
        {
            stateInThisThreads[st_id] = stateInThisThreads[st_id] + (d_sec * dstateInThisThreads[st_id]);
        }
        // ここまで終了　2021.9.1
        logBarrier += -logf(u[init_ID+1]+SCV->constraints[3])-logf(u[init_ID+2]+SCV->constraints[3])-logf(u[init_ID+3]+SCV->constraints[3]);
        logBarrier += -logf(SCV->constraints[3]-u[init_ID+1])-logf(SCV->constraints[3]-u[init_ID+2])-logf(SCV->constraints[3]-u[init_ID+3]);
        logBarrier += -logf(stateInThisThreads[6]+SCV->constraints[1])-logf(stateInThisThreads[7]+SCV->constraints[1])-logf(stateInThisThreads[8]+SCV->constraints[1]);
        logBarrier += -logf(SCV->constraints[1]-stateInThisThreads[6])-logf(SCV->constraints[1]-stateInThisThreads[7])-logf(SCV->constraints[1]-stateInThisThreads[8]);
        logBarrier += -logf(SCV->constraints[5]-u[init_ID])-logf(u[init_ID]);
        logBarrier += sRho * ((SCV->constraints[3]-SCV->constraints[2])+(SCV->constraints[1]-SCV->constraints[0])+(SCV->constraints[5]-SCV->constraints[4]));
        
        stageCost = SCV->weightMatrix[0] * (stateInThisThreads[0] - SCV->params[1]) * (stateInThisThreads[0] - SCV->params[1])
                    + SCV->weightMatrix[2] * (stateInThisThreads[2] - SCV->params[2]) * (stateInThisThreads[2] - SCV->params[2])
                    + SCV->weightMatrix[4] * (stateInThisThreads[4] - SCV->params[3]) * (stateInThisThreads[4] - SCV->params[3])
                    + SCV->weightMatrix[1] * stateInThisThreads[1] * stateInThisThreads[1] + SCV->weightMatrix[3] * stateInThisThreads[3] * stateInThisThreads[3]
                    + SCV->weightMatrix[5] * stateInThisThreads[5] * stateInThisThreads[5] + SCV->weightMatrix[6] * stateInThisThreads[6] * stateInThisThreads[6]
                    + SCV->weightMatrix[7] * stateInThisThreads[7] * stateInThisThreads[7] + SCV->weightMatrix[8] * stateInThisThreads[8] * stateInThisThreads[8]
                    + SCV->weightMatrix[9] * (u[init_ID] - SCV->params[0]) * (u[init_ID] - SCV->params[0]) + SCV->weightMatrix[10] * u[init_ID+1] * u[init_ID+1]
                    + SCV->weightMatrix[11] * u[init_ID+2] * u[init_ID+2] + SCV->weightMatrix[12] * u[init_ID+3] * u[init_ID+3];
        
        stageCost = stageCost / 2;
        
        /*if(t == HORIZON -1){
            stageCost += stateInThisThreads[0] * stateInThisThreads[0] * SCV->weightMatrix[0] + sinf(stateInThisThreads[1] / 2) * sinf(stateInThisThreads[1] / 2) * SCV->weightMatrix[1]
                        + stateInThisThreads[2] * stateInThisThreads[2] * SCV->weightMatrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * SCV->weightMatrix[3];
        }*/

        totalCost += stageCost + Rho * logBarrier;
        totalCost_Fit += stageCost;
        logBarrier = 0.0;
        stageCost = 0.0;
    }
    double KL_COST, S, lambda, HM_COST, HM;
    if(totalCost > 100){
        lambda =  10 * HORIZON * DIM_OF_INPUT;
    }else{
        lambda = mic * HORIZON * DIM_OF_INPUT;
    }
    
    HM = totalCost / (hdelta * HORIZON);
    S = totalCost / lambda;
    KL_COST = exp(-S);
    HM_COST = exp(-HM);
    __syncthreads();

    Info[id].W = KL_COST;
    Info[id].L = totalCost / NUM_OF_PARABOLOID_COEFFICIENT;
    Info[id].LF = totalCost_Fit / NUM_OF_PARABOLOID_COEFFICIENT;
    Info[id].WHM = HM_COST;
    cost_vec[id] = totalCost;
    int uIndex;
    for(int index = 0; index < HORIZON; index++){
        for(int id_input = 0; id_input < DIM_OF_INPUT; id_input++)
        {
            uIndex = index * DIM_OF_INPUT + id_input;
            if(uIndex % 4 == 0){
                Info[id].Input[id_input][index] = u[uIndex] / SCV->constraints[5];
            }else{
                Info[id].Input[id_input][index] = u[uIndex];
            }
            // Info[id].Input[id_input][index] = u[uIndex]; 
        }
    }
    __syncthreads();
}

__global__ void MCMPC_QuaternionBased_Quadrotor( SystemControlVariable *SCV, double var, curandState *randomSeed, double *mean, SampleInfo *Info, double *cost_vec)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq = id;
    int init_ID;
    double stageCost = 0.0;
    double totalCost = 0.0;
    double totalCost_Fit = 0.0;
    double logBarrier = 0.0;
    double u[HORIZON * DIM_OF_INPUT] = { };
    double current_guess[DIM_OF_INPUT] = { };
    double stateInThisThreads[DIM_OF_STATES] = { };
    double dstateInThisThreads[DIM_OF_STATES] = { };

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        stateInThisThreads[i] = SCV->state[i];
    }

    double d_sec = predictionInterval / HORIZON;
    for(int t = 0; t < HORIZON; t++)
    {
        init_ID = t * DIM_OF_INPUT;
        int index_input;
        // gen_multi_input(seq, randomSeed, )
        if(isnan(mean[t*DIM_OF_INPUT]) || isnan(mean[t*DIM_OF_INPUT+1]) || isnan(mean[t*DIM_OF_INPUT+2]) || isnan(mean[t*DIM_OF_INPUT+3])){
            for(int id_cg = 0; id_cg < DIM_OF_INPUT; id_cg++)
            {
                // current_guess[id_cg] = Info[0].Input[id_cg][t];
                index_input = t * DIM_OF_INPUT + id_cg;
                current_guess[id_cg] = mean[index_input];
            }
        }else{
            for(int id_cg = 0; id_cg < DIM_OF_INPUT; id_cg++)
            {
                index_input = t * DIM_OF_INPUT + id_cg;
                current_guess[id_cg] = mean[index_input];
            }
        }
        gen_multi_input(u, seq, randomSeed, t, current_guess, var);
        seq += NUM_OF_SAMPLES;
        /*if(id % 5000 == 0){
            printf("id::= %d, u[0]=:%lf, u[1] = %lf, u[2] = %lf, u[3] = %lf\n", id,  mean[init_ID], u[init_ID+1], u[init_ID+2], u[init_ID+3]);
        }*/
#ifdef InputSaturation
        if(u[init_ID] < SCV->constraints[4])
        {
            u[init_ID] = SCV->constraints[4] + zeta;
        }
        if(u[init_ID] > SCV->constraints[5])
        {
            u[init_ID] = SCV->constraints[5] - zeta;
        }
        for(int c_i = 1; c_i < DIM_OF_INPUT; c_i++)
        {
            if(u[init_ID+c_i] < SCV->constraints[2]){
                u[init_ID+c_i] = SCV->constraints[2] + zeta;
            }
            if(u[init_ID+c_i] > SCV->constraints[3]){
                u[init_ID+c_i] = SCV->constraints[3] - zeta;
            }
        }
#endif
        //まずは、(Δt = prediction interval(s) / HORIZON (step))の刻み幅でオイラー積分する方法
        // dynamics_ddot_Quadrotor(dstateInThisThreads, u[init_ID], u[init_ID + 1], u[init_ID + 2], u[init_ID + 3], stateInThisThreads, SCV); // get dx, ddx, dy, ddy, dz, ddz, dγ, dβ, dα
        dynamics_QuaternionBased_Quadrotor(dstateInThisThreads, u[init_ID], u[init_ID + 1], u[init_ID + 2], u[init_ID + 3], stateInThisThreads, SCV); // get dx, ddx, dy, ddy, dz, ddz, dγ, dβ, dα
        for(int st_id = 0; st_id < DIM_OF_STATES; st_id++)
        {
            stateInThisThreads[st_id] = stateInThisThreads[st_id] + (d_sec * dstateInThisThreads[st_id]);
        }
        // ここまで終了　2021.9.7
        // クォータニオンは正規化する必要がないよ
        // デカップリング入力で十分回転数の方も制限できるよ
        logBarrier += -logf(u[init_ID+1]+SCV->constraints[3])-logf(u[init_ID+2]+SCV->constraints[3])-logf(u[init_ID+3]+SCV->constraints[3]);
        logBarrier += -logf(SCV->constraints[3]-u[init_ID+1])-logf(SCV->constraints[3]-u[init_ID+2])-logf(SCV->constraints[3]-u[init_ID+3]);
        logBarrier += -logf(stateInThisThreads[6]+SCV->constraints[1])-logf(stateInThisThreads[7]+SCV->constraints[1])-logf(stateInThisThreads[8]+SCV->constraints[1]);
        logBarrier += -logf(SCV->constraints[1]-stateInThisThreads[6])-logf(SCV->constraints[1]-stateInThisThreads[7])-logf(SCV->constraints[1]-stateInThisThreads[8]);
        logBarrier += -logf(SCV->constraints[5]-u[init_ID])-logf(u[init_ID]);
        logBarrier += sRho * ((SCV->constraints[3]-SCV->constraints[2])+(SCV->constraints[1]-SCV->constraints[0])+(SCV->constraints[5]-SCV->constraints[4]));
        
        stageCost = SCV->weightMatrix[0] * (stateInThisThreads[0] - SCV->params[1]) * (stateInThisThreads[0] - SCV->params[1])
                    + SCV->weightMatrix[2] * (stateInThisThreads[2] - SCV->params[2]) * (stateInThisThreads[2] - SCV->params[2])
                    + SCV->weightMatrix[4] * (stateInThisThreads[4] - SCV->params[3]) * (stateInThisThreads[4] - SCV->params[3])
                    + SCV->weightMatrix[1] * stateInThisThreads[1] * stateInThisThreads[1] + SCV->weightMatrix[3] * stateInThisThreads[3] * stateInThisThreads[3]
                    + SCV->weightMatrix[5] * stateInThisThreads[5] * stateInThisThreads[5] + SCV->weightMatrix[6] * stateInThisThreads[6] * stateInThisThreads[6]
                    + SCV->weightMatrix[7] * stateInThisThreads[7] * stateInThisThreads[7] + SCV->weightMatrix[8] * stateInThisThreads[8] * stateInThisThreads[8]
                    + SCV->weightMatrix[9] * stateInThisThreads[10] * stateInThisThreads[10] + SCV->weightMatrix[10] * stateInThisThreads[11] * stateInThisThreads[11]
                    + SCV->weightMatrix[11] * stateInThisThreads[12] * stateInThisThreads[12]
                    + SCV->weightMatrix[12] * (u[init_ID] - SCV->params[0]) * (u[init_ID] - SCV->params[0]) + SCV->weightMatrix[13] * u[init_ID+1] * u[init_ID+1]
                    + SCV->weightMatrix[14] * u[init_ID+2] * u[init_ID+2] + SCV->weightMatrix[15] * u[init_ID+3] * u[init_ID+3];
        
        stageCost = stageCost / 2;
        
        /*if(t == HORIZON -1){
            stageCost += stateInThisThreads[0] * stateInThisThreads[0] * SCV->weightMatrix[0] + sinf(stateInThisThreads[1] / 2) * sinf(stateInThisThreads[1] / 2) * SCV->weightMatrix[1]
                        + stateInThisThreads[2] * stateInThisThreads[2] * SCV->weightMatrix[2] + stateInThisThreads[3] * stateInThisThreads[3] * SCV->weightMatrix[3];
        }*/

        totalCost += stageCost + Rho * logBarrier;
        totalCost_Fit += stageCost;
        logBarrier = 0.0;
        stageCost = 0.0;
    }
    double KL_COST, S, lambda, HM_COST, HM;
    if(totalCost > 100){
        lambda =  10 * HORIZON * DIM_OF_INPUT;
    }else{
        lambda = mic * HORIZON * DIM_OF_INPUT;
    }
    
    HM = totalCost / (hdelta * HORIZON);
    S = totalCost / lambda;
    KL_COST = exp(-S);
    HM_COST = exp(-HM);
    __syncthreads();

    Info[id].W = KL_COST;
    Info[id].L = totalCost / NUM_OF_PARABOLOID_COEFFICIENT;
    Info[id].LF = totalCost_Fit / NUM_OF_PARABOLOID_COEFFICIENT;
    Info[id].WHM = HM_COST;
    cost_vec[id] = totalCost;
    int uIndex;
    for(int index = 0; index < HORIZON; index++){
        for(int id_input = 0; id_input < DIM_OF_INPUT; id_input++)
        {
            uIndex = index * DIM_OF_INPUT + id_input;
            if(uIndex % 4 == 0){
                Info[id].Input[id_input][index] = u[uIndex] / SCV->constraints[5];
            }else{
                Info[id].Input[id_input][index] = u[uIndex];
            }
            // Info[id].Input[id_input][index] = u[uIndex]; 
        }
    }
    __syncthreads();
}