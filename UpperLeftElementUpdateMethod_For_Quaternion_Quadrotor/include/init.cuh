/*
 initialize all param for dynamical systems and setting of MCMPC
*/

#include <math.h>
#include "params.cuh"
#include "DataStructure.cuh"

void init_host_vector(double *params, double *states, double *constraints, double *matrix_elements);
void init_variables(SystemControlVariable *ret);
void set_IndexParams(IndexParams *gIdx );
unsigned int countBlocks(unsigned int a, unsigned int b);