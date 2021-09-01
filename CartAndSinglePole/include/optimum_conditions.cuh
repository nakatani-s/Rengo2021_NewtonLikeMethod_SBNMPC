/*
    optimum conditions consited with
    * cost function
    * dH/du 
    * dH/dx
    ...etc 
*/
#include <math.h>
#include <stdio.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"

double calc_cost_Cart_and_SinglePole(double *U, SystemControlVariable SCV);
double calc_tolerance_Cart_and_SinglePole(double *U, SystemControlVariable SCV);

void calc_OC_for_Cart_and_SinglePole_hostF(double *Ans, double *U, SystemControlVariable *SCV, Tolerance *Tol);
