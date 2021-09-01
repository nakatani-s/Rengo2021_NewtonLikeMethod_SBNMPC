/*
 DataStructure.cuh
*/

#include <curand_kernel.h>
#include "params.cuh"

#ifndef DATASTRUCTURE_CUH
#define DATASTRUCTURE_CUH

typedef struct{
    double L;
    double W;
    double WHM;
    double Input[HORIZON];
    double tolerance[HORIZON];
    double dHdu[HORIZON];
    double dHdx[DIM_OF_STATES][HORIZON];
}SampleInfo;

typedef struct{
    double tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT];
    double column_vector[NUM_OF_PARABOLOID_COEFFICIENT];
    double QplaneValue;
}QHP; // QHP := Quadratic Hyper Plane()


typedef struct{
    double params[DIM_OF_PARAMETERS];
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

#endif