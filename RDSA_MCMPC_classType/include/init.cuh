/*
This file includes definiton of all param initializing function for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
*/

#include <math.h>
#include "../rdsa_mcmpc_setupfile.cuh"
#include "DataStructures.cuh"

// void setSystemControllerVariable(SystemControllerVariable *initVariables);
void set_IdxParams(IndexParams *gIdx);