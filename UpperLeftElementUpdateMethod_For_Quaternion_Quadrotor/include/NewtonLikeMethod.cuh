/*
    NewtonLikeMethod.cuh cosist following part of Algorithm
    1. get tensor vector for generate regular matrix which is used in least squere mean
    2.
*/ 
#include<cuda.h>
#include<curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"


void NewtonLikeMethodInputSaturation(double *In, double Umax, double Umin);
void NewtonLikeMethodGetIterResult(SampleInfo *RetInfo, double costValue, double *InputSeq);

__global__ void NewtonLikeMethodCopyTensorVector(double *Mat, QHP *element, int Ydimention);
__global__ void NewtonLikeMethodGetTensorVectorNoIndex(QHP *Out, SampleInfo *Info);
__global__ void NewtonLikeMethodGetTensorVectorNormarizationed(QHP *Out, SampleInfo *In, int *indices, SystemControlVariable *SCV);

/* ------------ global functions are defined below -------------*/
__global__ void NewtonLikeMethodGetTensorVector(QHP *Out, SampleInfo *In, int *indices);
/* ------------- add 2021.9.2 ----------------*/
__global__ void NewtonLikeMethodGetTensorVectorTest(QHP *Out, SampleInfo *In, int *indices, mInputSystem mDim);
/* ------------- add 2021.10.6 ---------------*/ 
__global__ void NewtonLikeMethodGetTensorVectorPartial(QHP *Out, SampleInfo *In,double *LH, int *indices, const IndexParams *Idx, mInputSystem mDim);
__global__ void NewtonLikeMethodGetTensorVectorPartialDirect(QHP *Out, double *tensorA, double *tensorB, SampleInfo *In, double *LH, int *indices, const IndexParams *Idx, mInputSystem mDim);
__global__ void NewtonLikeMethodGetTensorVectorNoIndex(double *tensorL, QHP *In, const IndexParams *Idx);

__global__ void NewtonLikeMethodGenNormalizationMatrix(double *Mat, QHP *elements, int SAMPLE_SIZE, int Ydimention);
__global__ void NewtonLikeMethodGenNormalizationVector(double *Vec, QHP *elements, int SAMPLE_SIZE);
__global__ void NewtonLikeMethodGenNormalEquation(double *Mat, double *Vec, QHP *elements, int SAMPLE_SIZE, int Ydimention);

__global__ void NewtonLikeMethodGetRegularMatrix(double *Mat, QHP *element, int Sample_size);
__global__ void NewtonLikeMethodGetRegularMatrixTypeB(double *Mat, QHP *element, int Sample_size, int Ydimention);
__global__ void NewtonLikeMethodGetRegularVector(double *Vec, QHP *element, int Sample_size);

__global__ void NewtonLikeMethodGetRegularVectorPartial(double *Vec, QHP *element, int Sample_size);

// 最小二乗法の結果からヘシアンだけ取り出すための関数群
__global__ void NewtonLikeMethodGetHessianElements(double *HessElement, double *ansVec);
__global__ void NewtonLikeMethodGetHessianOriginal(double *Hessian, double *HessianElements);
/* ------------- add 2021.10.6 -------------*/ 
__global__ void NewtonLikeMethodGetBLCHessian(double *Hessian, double *blc_Hessian, double *HessianElement, const IndexParams *Idx);

__global__ void NewtonLikeMethodGetLowerTriangle(double *LowerTriangle, double *UpperTriangle);
__global__ void NewtonLikeMethodGetFullHessianLtoU(double *FullHessian, double *LowerTriangle);
__global__ void NewtonLikeMethodGetFullHessianUtoL(double *FullHessian, double *UpperTriangle);


// 最小二乗法の結果から勾配相当のベクトルを取り出すための関数
__global__ void NewtonLikeMethodGetGradient(double *Gradient, double *elements, int index);

// ベクトルのコピー，あとの処理を円滑に行うために
__global__ void NewtonLikeMethodCopyVector(double *Out, double *In);