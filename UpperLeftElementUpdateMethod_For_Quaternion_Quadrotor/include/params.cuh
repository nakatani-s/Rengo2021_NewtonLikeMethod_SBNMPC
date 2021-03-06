/*
params.cuh

*/
#include <math.h>
#ifndef PARAMS_CUH
#define PARAMS_CUH

#define InputSaturation
#define WRITE_MATRIX_INFORMATION
#define USING_QR_DECOMPOSITION
// #define READ_LQR_MATRIX
// #define USING_WEIGHTED_LEAST_SQUARES

#define SIM_TIME 1500
#define ITERATIONS_MAX 20
#define ITERATIONS 1
#define HORIZON 14
#define PART_HORIZON 9
// #define PART_HORIZON 8

#define DIM_OF_PARAMETERS 4
#define DIM_OF_SYSTEM_PARAM 11
#define DIM_OF_STATES 13
#define NUM_OF_CONSTRAINTS 6
#define DIM_OF_WEIGHT_MATRIX 16
#define DIM_OF_INPUT 4

#define NUM_OF_SAMPLES 10000
#define NUM_OF_ELITES 200
#define THREAD_PER_BLOCKS 10

const double predictionInterval = 0.98;
const double interval = 0.020; // control cycle for plant
const int NUM_OF_PARABOLOID_COEFFICIENT_L = 1653; //2145
// const int NUM_OF_PARABOLOID_COEFFICIENT_L = 2145;
const int NUM_OF_PARABOLOID_COEFFICIENT_S = 723; // (N^2 + N + 2*H + 2) / 2   N:=P_HORIZON*DIM_INPUT, H:=HORIZON*DIM_INPUT
// const int NUM_OF_PARABOLOID_COEFFICIENT_S = 885;
const int MAX_DIVISOR = 87;  // 87 Require divisor of "NUM_OF_PARABOLOID_COEFFICIENT" less than 1024
// const int MAX_DIVISOR = 37;  //Require divisor of "NUM_OF_PARABOLOID_COEFFICIENT" less than 1024
const int MAX_DIVISOR_S = 128;
const int addTermForLSM_L = 847; //847  3321 + 179 = 3500
// const int addTermForLSM_L = 299; // 3321 + 179 = 3500
const int addTermForLSM_S = 777;
// const int addTermForLSM_S = 1115;
// const int InputByHorizon = 80;
const double neighborVar = 0.35;
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