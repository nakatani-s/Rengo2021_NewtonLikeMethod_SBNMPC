/*
params.cuh

*/
#include <math.h>
#ifndef PARAMS_CUH
#define PARAMS_CUH

#define InputSaturation
#define WRITE_MATRIX_INFORMATION
#define USING_QR_DECOMPOSITION
// #define USING_WEIGHTED_LEAST_SQUARES

#define SIM_TIME 1000
#define ITERATIONS_MAX 20
#define ITERATIONS 2
#define HORIZON 18

#define DIM_OF_PARAMETERS 4
#define DIM_OF_STATES 9
#define NUM_OF_CONSTRAINTS 6
#define DIM_OF_WEIGHT_MATRIX 13
#define DIM_OF_INPUT 4

#define NUM_OF_SAMPLES 10000
#define NUM_OF_ELITES 100
#define THREAD_PER_BLOCKS 10

const double predictionInterval = 0.6;
const double interval = 0.020; // control cycle for plant
const int NUM_OF_PARABOLOID_COEFFICIENT = 2701;
const int MAX_DIVISOR = 9;  //Require divisor of "NUM_OF_PARABOLOID_COEFFICIENT" less than 1024 
const int addTermForLSM = 399; // 3321 + 179 = 3500
// const int InputByHorizon = 80;
const double neighborVar = 0.25;
const double variance = 2.0; // variance used for seaching initial solution by MCMPC with Geometric Cooling
const double Rho = 1e-6; // inverse constant values for Barrier term
const double sRho = 1e-4;
const double mic = 0.10;
const double hdelta = 0.10;
const double zeta = 1e-6;

// Parameters for cuBlas
const double alpha = 1.0;
const double m_alpha = -1.0;
const double beta = 0.0;

#endif