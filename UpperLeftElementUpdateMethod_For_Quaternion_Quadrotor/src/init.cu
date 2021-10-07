/*
 initialize all parameter for System model and cost function
*/ 
#include "../include/init.cuh"

void set_IndexParams(IndexParams *gIdx )
{
    int InputByHorizonLarge = DIM_OF_INPUT * HORIZON;
    int InputByHorizonSmall = DIM_OF_INPUT * PART_HORIZON;
    gIdx->InputByHorizonL = InputByHorizonLarge;
    gIdx->InputByHorizonS = InputByHorizonSmall;
    gIdx->dim_H_L = InputByHorizonLarge * InputByHorizonLarge;
    gIdx->dim_H_S = InputByHorizonSmall * InputByHorizonSmall;
    gIdx->num_UKPrm_QC_L = NUM_OF_PARABOLOID_COEFFICIENT_L;
    gIdx->num_UKPrm_QC_S = NUM_OF_PARABOLOID_COEFFICIENT_S;
    gIdx->pow_nUKPrm_QC_L = NUM_OF_PARABOLOID_COEFFICIENT_L * NUM_OF_PARABOLOID_COEFFICIENT_L;
    gIdx->pow_nUKPrm_QC_S = NUM_OF_PARABOLOID_COEFFICIENT_S * NUM_OF_PARABOLOID_COEFFICIENT_S;
    gIdx->num_UKPrm_H_L = NUM_OF_PARABOLOID_COEFFICIENT_L - (InputByHorizonLarge + 1);
    gIdx->num_UKPrm_H_S = NUM_OF_PARABOLOID_COEFFICIENT_S - (InputByHorizonLarge + 1);
    gIdx->sz_LCLsamples_L = NUM_OF_PARABOLOID_COEFFICIENT_L + addTermForLSM_L;
    gIdx->sz_LCLsamples_S = NUM_OF_PARABOLOID_COEFFICIENT_S + addTermForLSM_S;
} 
void init_params( double *a )
{
    // params for simple nonlinear systems
    // for Simple Nonlinear System
    /*a[0] = 1.0f;
    a[1] = 1.0f;*/

    // FOR CART AND POLE
    a[0] = 9.806650;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = 0.0;
}

void init_SystemParam( double *a)
{
    a[0] = 9.806650; // g
    a[1] = 150.0; // thrust equinalent = g
    a[2] = 230; // thrust max
    a[3] = a[2] * a[2]; // thrust_max^2
    a[4] = 5.0; // トルク最大値
    a[5] = 4.0e-6; // トルクレート
    a[6] = 0.0085; // Ixx イナーシャ
    a[7] = 0.008; // Iyy イナーシャ
    a[8] = 0.0165; // Izz
    a[9] = 1.0; // Mass
    a[10] = 0.5; // ROTOR_DISTANCE
}

void init_state( double *a )
{
    // FOR CART AND POLE
    a[0] = 1.0; //X
    a[1] = 0.0; //dX
    a[2] = 1.0; //Y
    a[3] = 0.0; //dY
    a[4] = 1.0; //Z
    a[5] = 0.0; //dZ
    a[6] = 0.0; //dγ
    a[7] = 0.0; //dβ
    a[8] = 0.0; //dα
    a[9] = 1.0; //cos(th/2) Quaternion1
    a[10] = 0.0; // Quaternion_2
    a[11] = 0.0; // Quaternion_3
    a[12] = 0.0; // Quaternion_4
}

void init_constraint( double *a )
{
    // FOR CONTROL CART AND POLE
    // For Quadric Fitting Superior Constraints parameters
    a[0] = -0.2; // for γ，β，α
    a[1] = 0.2; // for γ，β，α
    a[2] = -20.0; // u2 u3 u4
    a[3] = 20.0; // for u2 u3 u4
    a[4] = 0.0; // for u1
    // a[5] = 4 * 4.235; // for u1
    a[5] = 25.0;

    // For MC superior Parameter
    /*a[0] = -1.0f;
    a[1] = 1.0f;
    a[2] = -0.5f;
    a[3] = 0.5f;*/
}

void init_matrix( double *a )
{
    // FOR CAONTROL CART AND POLE
    // For Quadric Fitting Superior Weight Parameter
    /*a[0] = 3.0f;
    a[1] = 3.5f;
    a[2] = 0.0f;
    a[3] = 0.0f;
    a[4] = 1.0f;*/

    // For MC superior Parameter
    // Q
    a[0] = 10.0;
    a[1] = 1.0;
    a[2] = 10.0;
    a[3] = 1.0;
    a[4] = 20.0;
    a[5] = 2.0;
    a[6] = 10.0;
    a[7] = 10.0;
    a[8] = 10.0;
    a[9] = 100.0;
    a[10] = 100.0;
    a[11] = 100.0;

    // R
    a[12] = 0.1;
    a[13] = 0.1;
    a[14] = 0.1;
    a[15] = 0.1;
}

void init_host_vector(double *params, double *states, double *constraints, double *matrix_elements)
{
    init_params(params);
    init_state(states);
    init_constraint(constraints);
    init_matrix(matrix_elements);
}

void init_variables(SystemControlVariable *ret)
{
    double param[DIM_OF_PARAMETERS] , state[DIM_OF_STATES], constraints[NUM_OF_CONSTRAINTS], weightMatrix[DIM_OF_WEIGHT_MATRIX];
    double sparam[DIM_OF_SYSTEM_PARAM];
    init_params(param);
    init_state(state);
    init_constraint(constraints);
    init_matrix(weightMatrix);
    init_SystemParam( sparam );
    for(int i = 0; i < DIM_OF_PARAMETERS; i++)
    {
        ret->params[i] = param[i];
    }
    for(int i = 0; i < DIM_OF_SYSTEM_PARAM; i++)
    {
        ret->sparams[i] = sparam[i];
    }
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        ret->state[i] = state[i];
    }
    for(int i = 0; i < NUM_OF_CONSTRAINTS; i++)
    {
        ret->constraints[i] = constraints[i];
    }
    for(int i = 0; i < DIM_OF_WEIGHT_MATRIX; i++)
    {
        ret->weightMatrix[i] = weightMatrix[i];
    }
}
unsigned int countBlocks(unsigned int a, unsigned int b) {
	unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}