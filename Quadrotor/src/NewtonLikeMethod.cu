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
            Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
            Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L * In[indices[id]].WHM;
        #else
            Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0;
            Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L; 
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
                        for(int h = 0; h < DIM_OF_INPUT; h++){
        #ifdef USING_WEIGHTED_LEAST_SQUARES
                            Out[id].tensor_vector[next_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * sqrtf( In[indices[id]].WHM );
                            Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k] * In[indices[id]].WHM;
        #else
                            Out[id].tensor_vector[next_indices] = In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
                            Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[j][i] * In[indices[id]].Input[h][k];
        #endif
                            next_indices += 1;
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
                    Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[j][i] * In[indices[id]].WHM;
        #else
                    Out[id].tensor_vector[next_indices] = In[indices[id]].Input[j][i];
                    Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[j][i];
        #endif
                    next_indices += 1;
                }
            }
        #ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0 * sqrtf( In[indices[id]].WHM );
            Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L * In[indices[id]].WHM;
        #else
            Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0;
            Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L; 
        #endif
            __syncthreads();
            break;
        
        default:
            break;
    }
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
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0 * sqrtf( Info[id].WHM );
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = Info[id].L * Info[id].WHM;
#else
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0;
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = Info[id].L; 
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
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Ydimention + ix;
    Mat[idx] = 0.0; //initialization
    for(int index = 0; index < Sample_size; index++)
    {
        Mat[idx] += element[index].tensor_vector[ix] * element[index].tensor_vector[iy];
    }
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
            vector_id += (HORIZON - sum_a);
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