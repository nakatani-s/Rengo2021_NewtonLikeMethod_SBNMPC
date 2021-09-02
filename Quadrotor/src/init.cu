/*
 initialize all parameter for System model and cost function
*/ 
#include "../include/init.cuh"
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

void init_state( double *a )
{
    // FOR CART AND POLE
    a[0] = 1.0; //X
    a[1] = 0.0; //dX
    a[2] = 1.0; //Y
    a[3] = 0.0; //dY
    a[4] = 1.0; //Z
    a[5] = 0.0; //dZ
    a[6] = 0.0; //γ
    a[7] = 0.0; //β
    a[8] = 0.0; //α
}

void init_constraint( double *a )
{
    // FOR CONTROL CART AND POLE
    // For Quadric Fitting Superior Constraints parameters
    a[0] = -0.2; // for γ，β，α
    a[1] = 0.2; // for γ，β，α
    a[2] = -1.0; // u2 u3 u4
    a[3] = 1.0; // for u2 u3 u4
    a[4] = 0.0; // for u1
    a[5] = 11.0; // for u1

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
    a[0] = 10.0;
    a[1] = 1.0;
    a[2] = 10.0;
    a[3] = 1.0;
    a[4] = 10.0;
    a[5] = 1.0;
    a[6] = 1.0;
    a[7] = 1.0;
    a[8] = 1.0;
    
    a[9] = 0.1;
    a[10] = 0.1;
    a[11] = 0.1;
    a[12] = 0.1;
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
    init_params(param);
    init_state(state);
    init_constraint(constraints);
    init_matrix(weightMatrix);
    for(int i = 0; i < DIM_OF_PARAMETERS; i++)
    {
        ret->params[i] = param[i];
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