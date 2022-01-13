/*
This file includes functions relates MCMPC for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
*/

#include "../include/mcmpc.cuh"

unsigned int countBlocks(unsigned int a, unsigned int b)
{
    unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}

__host__ __device__ void compute_weighted_mean(double *out, IndexParams *Idx, int *indices, SampleInfo *info)
{
    double totalweight = 0.0;

    for(int i = 0; i < Idx->elite_sample_size; i++)
    {
        if(isnan(info[indices[i]].weight) || isinf(info[indices[i]].weight)){
            totalweight += 0.0;
        }else{
            totalweight += info[indices[i]].weight;
        }
    }
    for(int t = 0; t < Idx->InputByHorizon; t++){
        info[Idx->sample_size].input[t] = 0.0;
        for(int i = 0; i < Idx->elite_sample_size; i++)
        {
            if(isnan(info[indices[i]].weight) || isinf(info[indices[i]].weight))
            {
                info[Idx->sample_size].input[t] += 0.0;
            }else{
                info[Idx->sample_size].input[t] += (info[indices[i]].weight * info[indices[i]].input[t]) / totalweight;
            }
            if(isnan(info[Idx->sample_size].input[t]) || isinf(info[Idx->sample_size].input[t]))
            {
                out[t] = 0.0;
            }else{
                out[t] = info[Idx->sample_size].input[t];
            }
        }
    }
}
__global__ void setup_RandomSeed(curandState *st, int seed)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, id, 0, &st[id]);
}

__global__ void get_managed_indices(int *mng_indices_vec, int *thrust_indices_vec)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    mng_indices_vec[id] = thrust_indices_vec[id];
}

__device__ double gen_input(unsigned int id, curandState *state, double ave, double var)
{
    double ret;
    int local_ID = id;
    curandState localState;
    localState = state[local_ID];
    ret = curand_normal(&localState) * var + ave;
    return ret;
}

__global__ void calc_weighted_mean(double *out, IndexParams *Idx, int *indices, SampleInfo *info)
{
    double totalweight = 0.0;
    double *temp;
    temp = (double *)malloc(sizeof(double) * Idx->InputByHorizon);

    for(int i = 0; i < Idx->elite_sample_size; i++)
    {
        if(isnan(info[indices[i]].weight)){
            totalweight += 0.0; 
        }else{
            totalweight += info[indices[i]].weight;
        }
    }
    for(int t = 0; t < Idx->InputByHorizon; t++)
    {
        temp[t] = 0.0;
        for(int i = 0; i < Idx->elite_sample_size; i++){
            if(isnan(info[indices[i]].weight))
            {
                temp[t] += 0.0;
            }else{
                temp[t] += (info[indices[i]].weight * info[indices[i]].input[t]) / totalweight;
            }
        }
        if(isnan(temp[t]))
        {
            out[t] = 0.0;
        }else{
            out[t] = temp[t];
        }
    }
    free(temp);
}

__global__ void parallelSimForMC(double var, double *st, double *pr, double *re, double *co, double *we, double *mean, curandState *rndSeed, SampleInfo *info, IndexParams *Idx, double *cost_vec, int *indices)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq = id;
    int init_ID;
    // const int dim_input = CONTROLLER::HORIZON * OCP::DIM_OF_INPUT;

    double stageCost = 0.0;
    double totalCost = 0.0;
    // double totalCostF = 0.0;
    double logBarrier = 0.0;
    /*double *cg, *cu, *stateHere, *dstateHere;
    // u = (double *)malloc(sizeof(double) * Idx->horizon * Idx->dim_of_input);
    cg = (double *)malloc(sizeof(double) * Idx->dim_of_input);
    cu = (double *)malloc(sizeof(double) * Idx->dim_of_input);
    stateHere = (double *)malloc(sizeof(double) * Idx->dim_of_state);
    dstateHere = (double *)malloc(sizeof(double) * Idx->dim_of_state);
    // temp_u = (double *)malloc(sizeof(double) * Idx->dim_of_input * Idx->horizon);*/

    for(int i = 0; i< Idx->dim_of_state; i++)
    {
        info[id].dev_state[i] = st[i];
    }

    double d_sec = Idx->predict_interval / Idx->horizon;
    
    // 予測シミュレーション
    for(int t = 0; t < Idx->horizon; t++)
    {
        init_ID = t * Idx->dim_of_input;
        for(int id_i = 0; id_i < Idx->dim_of_input; id_i++)
        {
            if(isnan(mean[init_ID + id_i]))
            {
                info[id].dev_input[id_i] = 0.0;
            }else{
                info[id].dev_input[id_i] = mean[init_ID + id_i];
            }
            unsigned int h_seq = seq + id_i * (Idx->horizon * Idx->dim_of_input);
            info[id].dev_input[id_i] = gen_input(h_seq, rndSeed, info[id].dev_input[id_i], var); // ここで、入力のインデックスを回す実装の導入
        }
        seq += Idx->sample_size;
#ifdef InputSaturation
        input_constranint(info[id].dev_input.d_pointer(), co, Idx->zeta);
#endif
        // モデルを用いた予測シミュレーションの実行
        myDynamicModel(info[id].dev_dstate.d_pointer(), info[id].dev_input.d_pointer(), info[id].dev_state.d_pointer(), pr);
        // transition_Eular(stateHere, dstateHere, Idx->control_cycle, Idx->dim_of_state);
        transition_Eular(info[id].dev_state.d_pointer(), info[id].dev_dstate.d_pointer(), d_sec, Idx->dim_of_state);


        logBarrier = getBarrierTerm(info[id].dev_state.d_pointer(), info[id].dev_input.d_pointer(), co, Idx->sRho);
        stageCost = myStageCostFunction(info[id].dev_input.d_pointer(), info[id].dev_state.d_pointer(), re, we);

        // 入力列のコピー
        for(int i = 0; i < Idx->dim_of_input; i++)
        {
            info[id].input[init_ID + i] = info[id].dev_input[i];
        }
        /*if(id==1){
            printf("param[0] = %lf ** param[1] = %lf\n", pr[0], pr[1]);
            printf("input[0] = %lf input[1] = %lf\n", info[id].input[init_ID], info[id].input[init_ID+1]);
            printf("statge cost now = %lf **** logbarrier = %lf\n", stageCost, logBarrier);
        }*/
        totalCost += stageCost;
        
        if(isnan(logBarrier)){
            totalCost += 100;
        }else{
            totalCost += 1e-2*Idx->sRho*logBarrier;
        }
    }
    double KC, S, lambda;
    if(totalCost > Idx->horizon * Idx->dim_of_input){
        lambda = 10 * Idx->horizon * Idx->dim_of_input;
    }else{
        lambda = Idx->micro * Idx->horizon * Idx->dim_of_input;
    }
    S = totalCost / lambda;
    KC = exp(-S);
    // if(id < 10){ printf("%d sample has %lf cost value by %lf total cost\n", id, KC, totalCost);}
    // __syncthreads();
    info[id].cost = totalCost / Idx->FittingSampleSize;
    info[id].weight = KC;
    cost_vec[id] = totalCost;
    indices[id] = id;
    // __syncthreads();
    // free(u);
    /*free(cg);
    free(cu);
    free(stateHere);
    free(dstateHere);*/
    // free(temp_u);
}

double calc_cost(double *inputSeq, double *cs, double *prm, double *ref, double *cnstrnt, double *we, IndexParams *Idx)
{
    double stage_cost = 0.0;
    double total_cost = 0.0;
    double logBarrier;
    double *dstate, *cu, *hcs;
    dstate = (double *)malloc(sizeof(double) * Idx->dim_of_state);
    hcs = (double *)malloc(sizeof(double) * Idx->dim_of_state);
    cu = (double *)malloc(sizeof(double) * Idx->dim_of_input);
    int i_pointer = 0;
    double d_sec = Idx->predict_interval / Idx->horizon;
    // memcpy(hcs, cs, sizeof(double) * Idx->dim_of_state);
    for(int i = 0; i< Idx->dim_of_state; i++)
    {
        hcs[i] = cs[i];
    }
    for(int t = 0; t < Idx->horizon; t++)
    {
        i_pointer = t * Idx->dim_of_input;
        for(int i = 0; i < Idx->dim_of_input; i++)
        {
            cu[i] = inputSeq[i_pointer + i];
        }
        myDynamicModel(dstate, cu, hcs, prm);
        transition_Eular(hcs, dstate, d_sec, Idx->dim_of_state);
        logBarrier = getBarrierTerm(hcs, cu, cnstrnt, Idx->sRho);
        stage_cost = myStageCostFunction(cu, hcs, ref, we);
        
        // printf("statge cost now = %lf **** logbarrier = %lf\n", stage_cost, logBarrier);

        total_cost += stage_cost;

        if(isnan(logBarrier)){
            total_cost += 100;
        }else{
            total_cost += 1e-2*Idx->sRho*logBarrier;
        }
    }

    free(dstate);
    free(hcs);
    free(cu);
    return total_cost;
}

__global__ void devicePrnDebuggerSIF(double ts, SampleInfo *SIF)
{
    // unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    printf("time step :: %lf <====> cost value :: %lf\n", ts, SIF->cost);
}