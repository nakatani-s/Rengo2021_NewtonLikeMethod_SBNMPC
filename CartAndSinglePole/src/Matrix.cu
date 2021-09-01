/*
    Matrix.cu
*/
#include "../include/Matrix.cuh"

void printMatrix(int m, int n, double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %lf\n", name, row+1, col+1, Areg);
            //printf("%s[%d] = %lf\n", name, row + col*lda, Areg);
        }
    }
}

__global__ void MatrixSetUpLargeIdentityMatrix(double *IdMat, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * Ydimention + ix;

    if(ix == iy)
    {
        IdMat[id] = 1.0;
    }else{
        IdMat[id] = 0.0;
    }
    // __syncthreads();
}

__global__ void MatrixSetUpSmallIdentityMatrix(double *IdMat)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadIdx.x == blockIdx.x)
    {
        IdMat[id] = 1.0;
    }else{
        IdMat[id] = 0.0;
    }
    __syncthreads();
}

__global__ void MatrixMultiplyOperation(double *RetMat, double multiplyValue, double *OriginMat)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    RetMat[id] = multiplyValue * OriginMat[id];
    __syncthreads();
}