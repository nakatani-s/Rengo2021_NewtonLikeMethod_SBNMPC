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

__global__ void setup_RandomSeed(curandState *st, int seed)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, id, 0, &st[id]);
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

__global__ void parallelSimForMC(double var, double *st, double *pr, double *re, double *co, double *we, double *mean, curandState *rndSeed, SampleInfo *SIF, IndexParams *Idx, double *cost_vec)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seq = id;
    int init_ID;
    // const int dim_input = CONTROLLER::HORIZON * OCP::DIM_OF_INPUT;

    double stageCost = 0.0;
    double totalCost = 0.0;
    double totalCostF = 0.0;
    double logBarrier = 0.0;
    double *u, *cg, *cu, *stateHere, *dstateHere;
    // u = (double *)malloc(sizeof(double) * Idx->horizon * Idx->dim_of_input);
    cg = (double *)malloc(sizeof(double) * Idx->dim_of_input);
    cu = (double *)malloc(sizeof(double) * Idx->dim_of_input);
    stateHere = (double *)malloc(sizeof(double) * Idx->dim_of_state);
    dstateHere = (double *)malloc(sizeof(double) * Idx->dim_of_state);

    for(int i = 0; i< Idx->dim_of_state; i++)
    {
        stateHere[i] = st[i];
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
                cg[id_i] = 0.0;
            }else{
                cg[id_i] = mean[init_ID + id_i];
            }
            unsigned int h_seq = seq + id_i * (Idx->horizon * Idx->dim_of_input);
            cu[id_i] = gen_input(h_seq, rndSeed, cg[id_i], var); // ここで、入力のインデックスを回す実装の導入
        }
        seq += Idx->sample_size;
#ifdef InputSaturation
        input_constranint(cu, co, Idx->zeta);
#endif
        // モデルを用いた予測シミュレーションの実行
        myDynamicModel(dstateHere, cu, stateHere, pr);
        transition_Eular(stateHere, dstateHere, Idx->control_cycle, Idx->dim_of_state);

        logBarrier = getBarrierTerm(stateHere, cu, co, Idx->sRho);
        stageCost = myStageCostFunction(cu, stateHere, re, we);

        // 入力列のコピー
        for(int i = 0; i < Idx->dim_of_input; i++)
        {
            SIF[id].inputSeq[init_ID + i] = cu[i];
        }
    }

    free(u);
    free(cg);
    free(cu);
    free(stateHere);
    free(dstateHere);
}