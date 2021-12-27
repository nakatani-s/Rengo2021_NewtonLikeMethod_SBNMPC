/* 
------2021.12.21 start making-------------
*/

#include "../include/DataStructures.cuh"

// コンストラクタ
SampleInfo::SampleInfo(){
    cost = 0.0;
    weight = 0.0;
    int IBH = OCP::DIM_OF_INPUT * CONTROLLER::HORIZON;
    inputSeq = (double *)malloc(sizeof(double) * IBH);
}

SampleInfo::~SampleInfo(){
    free(inputSeq);
}

// コンストラクタ for QHP class
QHP::QHP()
{
    tensor_vector = (double *)malloc(sizeof(double) * OCP::DIM_OF_HESSIAN_ELEMENTS)
    column_vector = (double *)malloc(sizeof(double) * OCP::DIM_OF_HESSIAN_ELEMENTS);
}

QHP::~QHP()
{
    free(tensor_vector);
    free(column_vector);
}

/*SystemControlVariable::SystemControlVariable()
{
    params = (double *)malloc(sizeof(double) * OCP::DIM_OF_SYSTEM_PARAMS);
    reference = (double *)malloc(sizeof(double) * OCP::DIM_OF_REFERENCE);
    state = (double *)malloc(sizeof(double) * OCP::DIM_OF_STATES);
    constraints = (double *)malloc(sizeof(double) * OCP::DIM_OF_CONSTRAINTS);
    weightMatrix = (double *)malloc(sizeof(double) * OCP::DIM_OF_WEIGHT_MATRIX)
}

SystemControlVariable::~SystemControlVariable()
{
    free(params);
    free(reference);
    free(state);
    free(constraints);
    free(weightMatrix);
}*/