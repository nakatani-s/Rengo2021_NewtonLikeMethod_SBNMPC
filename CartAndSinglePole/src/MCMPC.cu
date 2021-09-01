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


void weighted_mean(double *Output, int num_elite, SampleInfo *hInfo)
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
}



__global__ void setup_kernel(curandState *state,int seed)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, id, 0, &state[id]);
}

__global__ void getEliteSampleInfo( SampleInfo *Elite, SampleInfo *All, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Elite[id].W = All[indices[id]].W;
    Elite[id].L = All[indices[id]].L;
    for(int i = 0; i < HORIZON; i++)
    {
        Elite[id].Input[i] = All[indices[id]].Input[i];
    }
    __syncthreads();
}

__device__ double gen_u(unsigned int id, curandState *state, double ave, double vr) {
    double u;
    curandState localState = state[id];
    u = curand_normal(&localState) * vr + ave;
    return u;
}



__global__ void MCMPC_Cart_and_SinglePole( SystemControlVariable *SCV, double var, curandState *randomSeed, double *mean, SampleInfo *Info, double *cost_vec)
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
}