/*

*/
#include "../include/NewtonLikeMethod.cuh"

void NewtonLikeMethodInputSaturation(double *In, double Umax, double Umin)
{
    for(int i = 0; i < HORIZON; i++)
    {
        if(In[i] > Umax)
        {
            In[i] = Umax -zeta;
        }
        if(In[i] < Umin)
        {
            In[i] = Umin + zeta;
        }
    }
}

/*void NewtonLikeMethodGetIterResult(SampleInfo *RetInfo, double costValue, double *InputSeq)
{
    double KL_COST, S, lambda, HM_COST, HM;
    lambda = mic * HORIZON;
    HM = costValue / (100 * HORIZON);
    S = costValue / lambda;
    KL_COST = exp(-S);
    HM_COST = exp(-HM);

    RetInfo->W = KL_COST;
    RetInfo->L = costValue / (100*HORIZON);
    RetInfo->WHM = HM_COST;
    // cost_vec[id] = totalCost;
    for(int index = 0; index < HORIZON; index++){
        RetInfo->Input[index] = InputSeq[index];
    }
}*/


/*__global__ void NewtonLikeMethodGetTensorVectorNormarizationed(QHP *Out, SampleInfo *In, int *indices, SystemControlVariable *SCV)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    int next_indices = 0;

    for(int i = 0; i < HORIZON; i++)
    {
        for(int j = i; j < HORIZON; j++)
        {
#ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i] * In[indices[id]].Input[j] * sqrtf( In[indices[id]].WHM );

            Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].Input[j] * In[indices[id]].WHM;

#else
            Out[id].tensor_vector[next_indices] = (In[indices[id]].Input[i] / SCV->constraints[1]) * (In[indices[id]].Input[j] / SCV->constraints[1]);

            Out[id].column_vector[next_indices] = In[indices[id]].L * (In[indices[id]].Input[i] / SCV->constraints[1]) * (In[indices[id]].Input[j] / SCV->constraints[1]);

#endif
            next_indices += 1;
        }
    }
    for(int i = 0; i < HORIZON; i++)
    {
#ifdef USING_WEIGHTED_LEAST_SQUARES
        Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i] * sqrtf( In[indices[id]].WHM );
        Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].WHM;

#else
        Out[id].tensor_vector[next_indices] = (In[indices[id]].Input[i] / SCV->constraints[1]);
        Out[id].column_vector[next_indices] = In[indices[id]].L * (In[indices[id]].Input[i] / SCV->constraints[1]);
#endif
        next_indices += 1;
    }

#ifdef USING_WEIGHTED_LEAST_SQUARES
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L * In[indices[id]].WHM;
#else
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0;
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L; 
#endif
    __syncthreads();
}*/

/*__global__ void NewtonLikeMethodGetTensorVector(QHP *Out, SampleInfo *In, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    int next_indices = 0;

    for(int i = 0; i < HORIZON; i++)
    {
        for(int j = i; j < HORIZON; j++)
        {
#ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i] * In[indices[id]].Input[j] * sqrtf( In[indices[id]].WHM );

            Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].Input[j] * In[indices[id]].WHM;

#else
            Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i] * In[indices[id]].Input[j];

            Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].Input[j];

#endif
            next_indices += 1;
        }
    }
    for(int i = 0; i < HORIZON; i++)
    {
#ifdef USING_WEIGHTED_LEAST_SQUARES
        Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i] * sqrtf( In[indices[id]].WHM );
        Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].WHM;

#else
        Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i];
        Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i];
#endif
        next_indices += 1;
    }

#ifdef USING_WEIGHTED_LEAST_SQUARES
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L * In[indices[id]].WHM;
#else
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0;
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L; 
#endif
    __syncthreads();
}*/

__global__ void NewtonLikeMethodGetTensorVectorTest(QHP *Out, SampleInfo *In, int *indices, mInputSystem mDim)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    int next_indices = 0;
    switch(mDim)
    {
        case SingleInput:
        
            for(int i = 0; i < HORIZON; i++)
            {
                for(int j = i; j < HORIZON; j++)
                {
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                    Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j] * sqrtf( In[indices[id]].WHM );
        
                    Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j] * In[indices[id]].WHM;
        
        #else
                    Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j];
        
                    Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j];
        
        #endif
                    next_indices += 1;
                }
            }
            for(int i = 0; i < HORIZON; i++)
            {
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i] * sqrtf( In[indices[id]].WHM );
                Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].WHM;
        
        #else
                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i];
                Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[0][i];
        #endif
                next_indices += 1;
            }
        
        #ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
            Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = In[indices[id]].L * In[indices[id]].WHM;
        #else
            Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = 1.0;
            Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = In[indices[id]].L; 
        #endif
            __syncthreads();
            break;
        
        case MultiInput:
    
            for(int i = 0; i < HORIZON; i++)
            {
                for(int j = 0; j < DIM_OF_INPUT; j++)
                {
                    for(int k = i; k < HORIZON; k++)
                    {
                        if(k == i){
                            for(int h = j; h < DIM_OF_INPUT; h++){
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * sqrtf( In[indices[id]].WHM );
                                Out[id].column_vector[next_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * In[indices[id]].WHM;
        #else
                                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                Out[id].column_vector[next_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
        #endif
                                next_indices += 1;
                            }
                        }else{
                            for(int h = 0; h < DIM_OF_INPUT; h++){
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * sqrtf( In[indices[id]].WHM );
                                Out[id].column_vector[next_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * In[indices[id]].WHM;
        #else
                                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                Out[id].column_vector[next_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
        #endif
                                next_indices += 1;
                            }
                        }
                        
                    }
                }
            }

            for(int i = 0; i < HORIZON; i++)
            {
                for(int j = 0; j < DIM_OF_INPUT; j++)
                {
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                    Out[id].tensor_vector[next_indices] = In[indices[id]].Input[j][i] * sqrtf( In[indices[id]].WHM );
                    Out[id].column_vector[next_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i] * In[indices[id]].WHM;
        #else
                    Out[id].tensor_vector[next_indices] = In[indices[id]].Input[j][i];
                    Out[id].column_vector[next_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i];
        #endif
                    next_indices += 1;
                }
            }
        #ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
            Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = In[indices[id]].LF * In[indices[id]].WHM;
        #else
            Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = 1.0;
            Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = In[indices[id]].LF;
        #endif
            __syncthreads();
            if(id == 0){
                printf("next_indices == %d\n",next_indices);
            }
            break;
        
        default:
            break;
    }
}

__global__ void NewtonLikeMethodGetTensorVectorPartial(QHP *Out, SampleInfo *In, double *LH, int *indices, const IndexParams *Idx, mInputSystem mDim)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    int next_indices = 0;
    int partial_blc_indices = 0;
    double TempDifferCostValue;
    switch(mDim)
    {
        case SingleInput:
        
            for(int i = 0; i < HORIZON; i++)
            {
                for(int j = i; j < HORIZON; j++)
                {
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                    Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j] * sqrtf( In[indices[id]].WHM );
        
                    Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j] * In[indices[id]].WHM;
        
        #else
                    Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j];
        
                    Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j];
        
        #endif
                    next_indices += 1;
                }
            }
            for(int i = 0; i < HORIZON; i++)
            {
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i] * sqrtf( In[indices[id]].WHM );
                Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].WHM;
        
        #else
                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i];
                Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[0][i];
        #endif
                next_indices += 1;
            }
        
        #ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[Idx->num_UKPrm_QC_S - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
            Out[id].column_vector[Idx->num_UKPrm_QC_S - 1] = In[indices[id]].L * In[indices[id]].WHM;
        #else
            Out[id].tensor_vector[Idx->num_UKPrm_QC_S - 1] = 1.0;
            Out[id].column_vector[Idx->num_UKPrm_QC_S - 1] = In[indices[id]].L; 
        #endif
            __syncthreads();
            break;
        
        case MultiInput:
        // tensor_vetorおよびcolumn_vectorのindexを見直す(今のままでは，固定係数の箇所分だけ空の配列となってしまう。)
        if(id == 0){
            printf("Idx->InputByHorizonS = %d next_indices == %d\n", Idx->InputByHorizonS,next_indices);
            printf("partial_blc_indices == %d\n", partial_blc_indices);
        }
            for(int i = 0; i < HORIZON; i++)
            {
                for(int j = 0; j < DIM_OF_INPUT; j++)
                {
                    for(int k = i; k < HORIZON; k++)
                    {
                        if(k < PART_HORIZON){
                            if(k == i){
                                for(int h = j; h < DIM_OF_INPUT; h++){
                         #ifdef USING_WEIGHTED_LEAST_SQUARES
                                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * sqrtf( In[indices[id]].WHM );
                                    Out[id].column_vector[partial_blc_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * In[indices[id]].WHM;
                        #else
                                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    Out[id].column_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                        #endif
                                    partial_blc_indices += 1;
                                    next_indices += 1;
                                }
                            }else{
                                for(int h = 0; h < DIM_OF_INPUT; h++){
                        #ifdef USING_WEIGHTED_LEAST_SQUARES
                                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * sqrtf( In[indices[id]].WHM );
                                    Out[id].column_vector[partial_blc_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * In[indices[id]].WHM;
                        #else
                                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    Out[id].column_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                        #endif
                                    partial_blc_indices += 1; 
                                    next_indices += 1;
                                }
                            }
                        }else{
                            if(k == i){
                                for(int h = j; h < DIM_OF_INPUT; h++){
                                    if(h == j){
                                        TempDifferCostValue += LH[next_indices] * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                        next_indices += 1;
                                    }else{
                                        TempDifferCostValue += LH[next_indices] * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                        next_indices += 1;
                                    }
                                }
                            }else{
                                for(int h = 0; h < DIM_OF_INPUT; h++){
                                    TempDifferCostValue += LH[next_indices] * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    next_indices += 1;
                                }
                            }
                        }
                    }      
                }
            }
            Out[id].DifferCostValue = TempDifferCostValue;
            Out[id].CostValue = In[indices[id]].LF;

            for(int i = 0; i < HORIZON; i++)
            {
                for(int j = 0; j < DIM_OF_INPUT; j++)
                {
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * sqrtf( In[indices[id]].WHM );
                    Out[id].column_vector[partial_blc_indices] =  In[indices[id]].Input[j][i] * In[indices[id]].WHM;
        #else
                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i];
                    Out[id].column_vector[partial_blc_indices] =  In[indices[id]].Input[j][i];
        #endif
                    // next_indices += 1;
                    partial_blc_indices += 1;
                }
            }
        #ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[Idx->num_UKPrm_QC_S - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
            Out[id].column_vector[Idx->num_UKPrm_QC_S - 1] = In[indices[id]].WHM;
        #else
            Out[id].tensor_vector[Idx->num_UKPrm_QC_S - 1] = 1.0;
            Out[id].column_vector[Idx->num_UKPrm_QC_S - 1] = 1.0;
        #endif
            __syncthreads();
            if(id == 0){
                printf("next_indices == %d\n",next_indices);
                printf("partial_blc_indices == %d\n", partial_blc_indices);
            }
            break;
        
        default:
            break;
    }
}

__global__ void NewtonLikeMethodGetTensorVectorPartialDirect(QHP *Out, double *tensorA, double *tensorB, SampleInfo *In, double *LH, int *indices, const IndexParams *Idx, mInputSystem mDim)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    int next_indices = 0;
    int partial_blc_indices = 0;
    double TempDifferCostValue;
    switch(mDim)
    {
        case SingleInput:
        
            for(int i = 0; i < HORIZON; i++)
            {
                for(int j = i; j < HORIZON; j++)
                {
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                    Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j] * sqrtf( In[indices[id]].WHM );
        
                    Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j] * In[indices[id]].WHM;
        
        #else
                    Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j];
        
                    Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[0][i] * In[indices[id]].Input[0][j];
        
        #endif
                    next_indices += 1;
                }
            }
            for(int i = 0; i < HORIZON; i++)
            {
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i] * sqrtf( In[indices[id]].WHM );
                Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].WHM;
        
        #else
                Out[id].tensor_vector[next_indices] = In[indices[id]].Input[0][i];
                Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[0][i];
        #endif
                next_indices += 1;
            }
        
        #ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[Idx->num_UKPrm_QC_S - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
            Out[id].column_vector[Idx->num_UKPrm_QC_S - 1] = In[indices[id]].L * In[indices[id]].WHM;
        #else
            Out[id].tensor_vector[Idx->num_UKPrm_QC_S - 1] = 1.0;
            Out[id].column_vector[Idx->num_UKPrm_QC_S - 1] = In[indices[id]].L; 
        #endif
            __syncthreads();
            break;
        
        case MultiInput:
        // tensor_vetorおよびcolumn_vectorのindexを見直す(今のままでは，固定係数の箇所分だけ空の配列となってしまう。)
        if(id == 0){
            printf("Idx->InputByHorizonS = %d next_indices == %d\n", Idx->InputByHorizonS,next_indices);
            printf("partial_blc_indices == %d\n", partial_blc_indices);
        }
            for(int i = 0; i < HORIZON; i++)
            {
                for(int j = 0; j < DIM_OF_INPUT; j++)
                {
                    for(int k = i; k < HORIZON; k++)
                    {
                        if(k < PART_HORIZON){
                            if(k == i){
                                for(int h = j; h < DIM_OF_INPUT; h++){
                         #ifdef USING_WEIGHTED_LEAST_SQUARES
                                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * sqrtf( In[indices[id]].WHM );
                                    Out[id].column_vector[partial_blc_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * In[indices[id]].WHM;
                        #else
                                    tensorA[id * Idx->num_UKPrm_QC_S + partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    tensorB[id * Idx->num_UKPrm_QC_S + partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    Out[id].column_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                        #endif
                                    partial_blc_indices += 1;
                                    next_indices += 1;
                                }
                            }else{
                                for(int h = 0; h < DIM_OF_INPUT; h++){
                        #ifdef USING_WEIGHTED_LEAST_SQUARES
                                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * sqrtf( In[indices[id]].WHM );
                                    Out[id].column_vector[partial_blc_indices] = In[indices[id]].LF * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * In[indices[id]].WHM;
                        #else
                                    tensorA[id * Idx->num_UKPrm_QC_S + partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    tensorB[id * Idx->num_UKPrm_QC_S + partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    Out[id].column_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                        #endif
                                    partial_blc_indices += 1; 
                                    next_indices += 1;
                                }
                            }
                        }else{
                            if(k == i){
                                for(int h = j; h < DIM_OF_INPUT; h++){
                                    if(h == j){
                                        TempDifferCostValue += LH[next_indices] * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                        next_indices += 1;
                                    }else{
                                        TempDifferCostValue += LH[next_indices] * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                        next_indices += 1;
                                    }
                                }
                            }else{
                                for(int h = 0; h < DIM_OF_INPUT; h++){
                                    TempDifferCostValue += LH[next_indices] * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                                    next_indices += 1;
                                }
                            }
                        }
                    }      
                }
            }
            Out[id].DifferCostValue = TempDifferCostValue;
            Out[id].CostValue = In[indices[id]].LF;
            // tensorL[id] = Out[id].CostValue - TempDifferCostValue;
            for(int i = 0; i < HORIZON; i++)
            {
                for(int j = 0; j < DIM_OF_INPUT; j++)
                {
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i] * sqrtf( In[indices[id]].WHM );
                    Out[id].column_vector[partial_blc_indices] =  In[indices[id]].Input[j][i] * In[indices[id]].WHM;
        #else
                    tensorA[id * Idx->num_UKPrm_QC_S + partial_blc_indices] = In[indices[id]].Input[j][i];
                    tensorB[id * Idx->num_UKPrm_QC_S + partial_blc_indices] = In[indices[id]].Input[j][i];
                    Out[id].tensor_vector[partial_blc_indices] = In[indices[id]].Input[j][i];
                    Out[id].column_vector[partial_blc_indices] = In[indices[id]].Input[j][i];
        #endif
                    // next_indices += 1;
                    partial_blc_indices += 1;
                }
            }
        #ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[Idx->num_UKPrm_QC_S - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
            Out[id].column_vector[Idx->num_UKPrm_QC_S - 1] = In[indices[id]].WHM;
        #else
            tensorA[id * Idx->num_UKPrm_QC_S + partial_blc_indices] = 1.0;
            tensorB[id * Idx->num_UKPrm_QC_S + partial_blc_indices] = 1.0;
            Out[id].tensor_vector[Idx->num_UKPrm_QC_S - 1] = 1.0;
            Out[id].column_vector[Idx->num_UKPrm_QC_S - 1] = 1.0;
        #endif
            // tensorL[id] = In[indices[id]].LF - TempDifferCostValue;
            __syncthreads();
            // tensorL[id] = Out[id].CostValue - TempDifferCostValue;
            if(id == 0){
                printf("next_indices == %d\n",next_indices);
                printf("partial_blc_indices == %d\n", partial_blc_indices);
            }
            break;
        
        default:
            break;
    }
}
__global__ void NewtonLikeMethodGetTensorVectorNoIndex(double *tensorL, QHP *In, const IndexParams *Idx)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(Idx->sz_LCLsamples_S <= id)
        return;
    tensorL[id] = In[id].CostValue - In[id].DifferCostValue;
    __syncthreads(); 
}

__global__ void NewtonLikeMethodGetTensorVectorNoIndex(QHP *Out, SampleInfo *Info)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    int next_indices = 0;

    for(int i = 0; i < HORIZON; i++)
    {
        for(int j = i; j < HORIZON; j++)
        {
#ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[next_indices] = Info[id].Input[0][i] * Info[id].Input[0][j] * sqrtf( Info[id].WHM );

            Out[id].column_vector[next_indices] = Info[id].L * Info[id].Input[0][i] * Info[id].Input[0][j] * Info[id].WHM;

#else
            Out[id].tensor_vector[next_indices] = Info[id].Input[0][i] * Info[id].Input[0][j];

            Out[id].column_vector[next_indices] = Info[id].L * Info[id].Input[0][i] * Info[id].Input[0][j];

#endif
            next_indices += 1;
        }
    }
    for(int i = 0; i < HORIZON; i++)
    {
#ifdef USING_WEIGHTED_LEAST_SQUARES
        Out[id].tensor_vector[next_indices] = Info[id].Input[0][i] * sqrtf( Info[id].WHM );
        Out[id].column_vector[next_indices] = Info[id].L * Info[id].Input[0][i] * Info[id].WHM;

#else
        Out[id].tensor_vector[next_indices] = Info[id].Input[0][i];
        Out[id].column_vector[next_indices] = Info[id].L * Info[id].Input[0][i];
#endif
        next_indices += 1;
    }

#ifdef USING_WEIGHTED_LEAST_SQUARES
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = 1.0 * sqrtf( Info[id].WHM );
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = Info[id].L * Info[id].WHM;
#else
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = 1.0;
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT_L - 1] = Info[id].L; 
#endif
    __syncthreads( );
}


// 正規方程式に関するベクトルGを計算するやつ
__global__ void NewtonLikeMethodGenNormalizationMatrix(double *Mat, QHP *elements, int SAMPLE_SIZE, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Ydimention + ix;

    Mat[idx] = 0.0; //initialization
    for(int index = 0; index < SAMPLE_SIZE; index++)
    {
        Mat[idx] += elements[index].tensor_vector[ix] * elements[index].tensor_vector[iy];
    }
    __syncthreads();
}


__global__ void NewtonLikeMethodGenNormalizationVector(double *Vec, QHP *elements, int SAMPLE_SIZE)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Vec[id] = 0.0; //initialization
    for(int index = 0; index < SAMPLE_SIZE; index++)
    {
        Vec[id] += elements[index].column_vector[id];
    }
    __syncthreads( );
}

__global__ void NewtonLikeMethodGenNormalEquation(double *Mat, double *Vec, QHP *elements, int SAMPLE_SIZE, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Ydimention + ix;
    Mat[idx] = 0.0;
    if(idx < Ydimention)
    {
        for(int index = 0; index < SAMPLE_SIZE; index++)
        {
            Mat[idx] += elements[index].tensor_vector[ix] * elements[index].tensor_vector[iy];
            Vec[idx] += elements[index].column_vector[idx];
        }
    }else{
        for(int index = 0; index < SAMPLE_SIZE; index++)
        {
            Mat[idx] += elements[index].tensor_vector[ix] * elements[index].tensor_vector[iy];
        }
    }
}

__global__ void NewtonLikeMethodGetRegularMatrix(double *Mat, QHP *element, int Sample_size)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Mat[id] = 0.0;
    for(int index = 0; index < Sample_size; index++)
    {
        Mat[id] += element[index].tensor_vector[threadIdx.x] * element[index].tensor_vector[blockIdx.x];
    }
    __syncthreads();
}

__global__ void NewtonLikeMethodGetRegularMatrixTypeB(double *Mat, QHP *element, int Sample_size, int Ydimention)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    // unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Ydimention + ix;
    Mat[idx] = 0.0; //initialization
    for(int index = 0; index < Sample_size; index++)
    {
        Mat[idx] += element[index].tensor_vector[ix] * element[index].tensor_vector[iy];
    }
    __syncthreads();

}

__global__ void NewtonLikeMethodCopyTensorVector(double *Mat, QHP *element, int Ydimention)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    // unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Ydimention + ix;
    Mat[idx] = element[ix].tensor_vector[iy];
    /*if(idx <= Ydimention * HORIZON ){
        if(idx % Ydimention == 0){
            printf("element[%d].tensor_vector[%d] = %lf\n", ix, iy, element[ix].tensor_vector[iy]);
        }
    }*/
    __syncthreads();
}

__global__ void NewtonLikeMethodGetRegularVector(double *Vec, QHP *element, int Sample_size)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Vec[id] = 0.0;
    for(int index = 0; index < Sample_size; index++)
    {
        Vec[id] += element[index].column_vector[id];
    }
    __syncthreads();
}

__global__ void NewtonLikeMethodGetRegularVectorPartial(double *Vec, QHP *element, int Sample_size)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Vec[id] = 0.0;
    for(int index = 0; index < Sample_size; index++)
    {
        Vec[id] +=  (element[index].CostValue - element[index].DifferCostValue) * element[index].column_vector[id];
    }
    __syncthreads();
}


__global__ void NewtonLikeMethodGetHessianElements(double *HessElement, double *ansVec)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    HessElement[id] = ansVec[id];
    // __syncthreads( ); 
}

__global__ void NewtonLikeMethodGetHessianOriginal(double *Hessian, double *HessianElements)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    double temp_here;

    int vector_id = blockIdx.x;
    if(threadIdx.x <= blockIdx.x)
    {
        for(int t_id = 0; t_id < threadIdx.x; t_id++)
        {
            int sum_a = t_id + 1;
            vector_id += (HORIZON * DIM_OF_INPUT - sum_a);
            // vector_id += (HORIZON - sum_a);
        }
        temp_here = HessianElements[vector_id];
    }else{
        temp_here = 0.0;
    }

    if(threadIdx.x != blockIdx.x)
    {
        if(isnan(temp_here))
        {
            Hessian[id] = Hessian[id];
        }else{
            Hessian[id] = temp_here / 2;
        }
    }else{
        if(isnan(temp_here))
        {
            Hessian[id] = 1.0;
        }else{
            Hessian[id] = temp_here;
        }
    }
    // __syncthreads();
}

__global__ void NewtonLikeMethodGetBLCHessian(double *Hessian, double *blc_Hessian, double *HessianElement, const IndexParams *Idx)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    double temp_here;

    int vector_id = blockIdx.x;
    if(threadIdx.x <= blockIdx.x){
        if(threadIdx.x < Idx->InputByHorizonS && blockIdx.x < Idx->InputByHorizonS )
        {
            for(int t_id = 0; t_id < threadIdx.x; t_id++)
            {
                int sum_id = t_id + 1;
                vector_id += (Idx->InputByHorizonS - sum_id);
            }
            temp_here = blc_Hessian[vector_id];
        }else{
            for(int t_id = 0; t_id < threadIdx.x; t_id++)
            {
                int sum_id = t_id + 1;
                vector_id +=(Idx->InputByHorizonL - sum_id);
            }
            temp_here = HessianElement[vector_id];
        }
    }else{
        temp_here = 0.0;
    }
    
    if(threadIdx.x != blockIdx.x)
    {
        if(isnan(temp_here))
        {
            Hessian[id] = Hessian[id];
        }else{
            Hessian[id] = temp_here / 2;
        }
    }else{
        if(isnan(temp_here))
        {
            Hessian[id] = 1.0;
        }else{
            Hessian[id] = temp_here;
        }
    }
    // __syncthreads();
}

__global__ void NewtonLikeMethodGetLowerTriangle(double *LowerTriangle, double *UpperTriangle)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int t_id = blockIdx.x + threadIdx.x * blockDim.x;

    LowerTriangle[id] = UpperTriangle[t_id];
}

__global__ void NewtonLikeMethodGetFullHessian(double *FullHessian, double *LowerTriangle)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    //この条件文が反対の可能性もある。2021.7.16(NAK)
    if(blockIdx.x < threadIdx.x )
    {
        if((!FullHessian[id] == LowerTriangle[id]))
        {
            FullHessian[id] = LowerTriangle[id];
        }
    }
}

__global__ void NewtonLikeMethodGetFullHessianUtoL(double *FullHessian, double *UpperTriangle)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(blockIdx.x > threadIdx.x)
    {
        if(!(FullHessian[id] == UpperTriangle[id]))
        {
            FullHessian[id] = UpperTriangle[id];
        }
    }
}

__global__ void NewtonLikeMethodGetGradient(double *Gradient, double *elements, int index)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Gradient[id] = elements[index + id];
    __syncthreads( );
}

__global__ void NewtonLikeMethodCopyVector(double *Out, double *In)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Out[id] = In[id];
    __syncthreads( );
}