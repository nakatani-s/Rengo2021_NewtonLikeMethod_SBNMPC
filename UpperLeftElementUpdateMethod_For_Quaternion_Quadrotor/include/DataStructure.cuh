/*
 DataStructure.cuh
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
    double Input[DIM_OF_INPUT][HORIZON];
    double tolerance[HORIZON];
    double dHdu[HORIZON];
    double dHdx[DIM_OF_STATES][HORIZON];
}SampleInfo;

typedef struct{
    double tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT_L];
    double column_vector[NUM_OF_PARABOLOID_COEFFICIENT_L];
    double QplaneValue;
    double DifferCostValue;
}QHP; // QHP := Quadratic Hyper Plane()


typedef struct{
    double params[DIM_OF_PARAMETERS];
    double sparams[DIM_OF_SYSTEM_PARAM];
    double state[DIM_OF_STATES];
    double constraints[NUM_OF_CONSTRAINTS];
    double weightMatrix[DIM_OF_WEIGHT_MATRIX];
}SystemControlVariable;

typedef struct{
    double Input[HORIZON];
}InputData;

typedef struct{
    double Input[DIM_OF_INPUT];
    double lambda[DIM_OF_STATES];
    double dHdu[DIM_OF_INPUT];
    double state[DIM_OF_STATES];
    double dstate[DIM_OF_STATES];
}Tolerance;

typedef struct{
    unsigned int InputByHorizonL;
    unsigned int InputByHorizonS;
    unsigned int dim_H_L;
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

enum mInputSystem{
    SingleInput, MultiInput
};

#endif