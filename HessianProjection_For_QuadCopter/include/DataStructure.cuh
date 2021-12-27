/*
This file includes definition of all structure variables for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
*/

#include <curand_kernel.h>
#include "params.cuh"

#ifndef DATASTRUCTURE_CUH
#define DATASTRUCTURE_CUH

typedef struct{
    double L;
    double LF;
    double W;
    double WHM;
    double IT_weight;
    double Input[DIM_OF_INPUT][HORIZON];
    // double tolerance[HORIZON];
    // double dHdu[HORIZON];
    // double dHdx[DIM_OF_STATES][HORIZON];
}SampleInfo;


typedef struct{
    double tensor_vector[NUM_OF_HESSIAN_ELEMENTS];
    double column_vector[NUM_OF_HESSIAN_ELEMENTS];
    double QplaneValue;
    double CostValue;
    double DifferCostValue;
}QHP; 

typedef struct{
    double params[DIM_OF_PARAMETERS];
    double sparams[DIM_OF_SYSTEM_PARAM];
    double state[DIM_OF_STATES];
    double constraints[NUM_OF_CONSTRAINTS];
    double weightMatrix[DIM_OF_WEIGHT_MATRIX];
}SystemControlVariable;

typedef struct{
    unsigned int InputByHorizon;
    unsigned int HessianSize;
    unsigned int size_HessElements;
    unsigned int pow_size_HessElements;
    unsigned int InputByHorizonS;
    unsigned int HessianSize;
    unsigned int dim_H_S;
    unsigned int num_UKPrm_QC_L; //旧: numUnknownParamQHP
    unsigned int num_UKPrm_QC_S;
    unsigned int pow_nUKPrm_QC_L;
    unsigned int pow_nUKPrm_QC_S;
    unsigned int num_UKPrm_H_L;
    unsigned int num_UKPrm_H_S;
    unsigned int sz_LCLsamples_L;
    unsigned int sz_LCLsamples_S;    
}IndexParams;

#endif