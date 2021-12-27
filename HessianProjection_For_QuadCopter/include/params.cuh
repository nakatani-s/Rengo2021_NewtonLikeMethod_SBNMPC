/*
This file includes control & system parameters for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
*/
#include <math.h>
#ifndef PARAMS_CUH
#define PARAMS_CUH

/* Control specific parameters */

/* Problem setting */
#define SIM_TIME 1500

/* Optimization problem setting */
#define ITERATIONS_MAX 20
#define ITERATIONS 1
#define HORIZON 10

/* System parameters */
#define DIM_OF_PARAMETERS 4
#define DIM_OF_SYSTEM_PARAM 11
#define DIM_OF_STATES 13
#define NUM_OF_CONSTRAINTS 6
#define DIM_OF_WEIGHT_MATRIX 16
#define DIM_OF_INPUT 4

/* Controller parameters */
#define NUM_OF_SAMPLES 10000
#define NUM_OF_ELITES 200
#define THREAD_PER_BLOCKS 10

const double predictionInterval = 0.90;
const double interval = 0.020;

const int max_divisor = 50;
const int NUM_OF_HESSIAN_ELEMENTS = 820;


#endif 