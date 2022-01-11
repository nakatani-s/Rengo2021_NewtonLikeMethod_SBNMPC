/*
------2021.12.21 making start
*/
#include <math.h>
#include <cuda.h>
#include "myControllerSrc.cuh"
// #include "mcmpc.cuh"
#ifndef MYCONTROLLER_CUH
#define MYCONTROLLER_CUH

struct OCP
{
    /* Problem setting */ 
    static const int SIM_STEPS;

    /* System parameters */
    static const int DIM_OF_REFERENCE;
    static const int DIM_OF_SYSTEM_PARAMS;
    static const int DIM_OF_SYSTEM_STATE;
    static const int DIM_OF_INPUT;

    static const int DIM_OF_CONSTRAINTS; //制約の数 (例：u_min < u < umax, x_max < x < x_max なら4)
    static const int DIM_OF_WEIGHT_MATRIX; //コスト関数の重み（Q,R）の要素数
    // static const int DIM_OF_HESSIAN;
    // static const int DIM_OF_HESSIAN_ELEMENTS;

    const double *initial_state;
};

struct CONTROLLER
{
    /* Optimization problem setting */
    static const int ITERATIONS_MAX;
    static const int ITERATIONS;
    static const int HORIZON;

    /* Controller parameters */ 
    static const int NUM_OF_SAMPLES;
    static const int THREAD_PER_BLOCKS;
    static const int NUM_OF_ELITE_SAMPLES;

    static const double PREDICTION_INTERVAL; // 予測区間（s）
    static const double CONTROL_CYCLE; //制御周期
    static const double SIGMA;
    static const int MAX_DIVISOR;
    // static const int NUM_OF_HESSIAN_ELEMENT;

    static const double c_rate;
    static const double zeta;
    static const double sRho;
    static const double Micro;
};
__host__ __device__ void input_constranint(double *u, double *constraints, double zeta);
__host__ __device__ double getBarrierTerm(double *st, double *u, double *co, double sRho);
__host__ __device__ void myDynamicModel(double *dstate, double *u, double *st, double *param);
__host__ __device__ double myStageCostFunction(double *u, double *st, double *reference, double *weightMatrix);
#endif
