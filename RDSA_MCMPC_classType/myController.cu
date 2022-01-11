/*
-----------2021.12.21 makin start 
*/

#include "include/myController.cuh"
// #include "include/mcmpc.cuh"
// #include "include/init.cuh"

/* 以下のパラメータは必ず設定して下さい。 */ 
// The following paameters are mandatory!
const int OCP::SIM_STEPS = 15;

const int OCP::DIM_OF_REFERENCE = 4;
const int OCP::DIM_OF_SYSTEM_PARAMS = 11;
const int OCP::DIM_OF_SYSTEM_STATE = 13;
const int OCP::DIM_OF_INPUT = 4;

const int OCP::DIM_OF_CONSTRAINTS = 6;
const int OCP::DIM_OF_WEIGHT_MATRIX = 16;

const int CONTROLLER::NUM_OF_SAMPLES = 10;
const int CONTROLLER::NUM_OF_ELITE_SAMPLES = 10;
const double CONTROLLER::PREDICTION_INTERVAL = 0.90;
const double CONTROLLER::CONTROL_CYCLE = 0.020;
const int CONTROLLER::THREAD_PER_BLOCKS = 10;
const int CONTROLLER::ITERATIONS_MAX = 10;
const int CONTROLLER::ITERATIONS = 1;
const int CONTROLLER::HORIZON = 5;

const double CONTROLLER::SIGMA = 1.0;
const int CONTROLLER::MAX_DIVISOR = 50;
// const int CONTROLLER::NUM_OF_HESSIAN_ELEMENT = 820;

// 以下の値は、推奨値
const double CONTROLLER::c_rate = 0.95; //デフォルトはこの値
const double CONTROLLER::zeta = 0.05; // デフォルトはこの値
const double CONTROLLER::sRho = 1e-4;
const double CONTROLLER::Micro = 1e-3;

/* ここは変更しない */
// const int OCP::DIM_OF_HESSIAN = OCP::DIM_OF_INPUT * CONTROLLER::HORIZON;
// const int numerator_temp = OCP::DIM_OF_HESSIAN * (OCP::DIM_OF_HESSIAN + 1); 
// const int OCP::DIM_OF_HESSIAN_ELEMENTS = (int)(numerator_temp/2);

/*void setSystemControllerVariable(SystemControllerVariable *initVariables)
{
    double 
}*/

__device__ void input_constranint(double *u, double *constraints, double zeta)
{
    check_constraint(u[0], constraints[0], constraints[1], zeta);
    for(int i = 1; i < 4; i++)
    {
        check_constraint(u[i], constraints[2], constraints[3], zeta);
    }
}

__device__ double getBarrierTerm(double *st, double *u, double *co, double sRho)
{
    double ret = 0.0;
    //ここから自身の制御問題に合わせて記載
    for(int i = 1; i < 4; i++)
    {
        ret += barrierConsraint(u[i], co[2], co[3], sRho);
    }
    ret += barrierConsraint(u[0], co[4], co[5], sRho);
    for(int i = 6; i < 9; i++)
    {
        ret += barrierConsraint(st[i], co[0], co[1], sRho);
    }
    return ret;
}

// *dstate に dot{x} = f(x,u,p)のdot{x}を代入する関数を記述
__host__ __device__ void myDynamicModel(double *dstate, double *u, double *currentState, double *param)
{
    // 下記は、クアッドコプター（姿勢角を四元数で表現）のモデルの例
    double o[10] = { };
    double c_state[13] = { };
    o[0] = param[9] * param[3];
    o[1] = param[1] + u[0] - u[2] + u[3];  // u1 = u[0], u2 = u[1], u3 = u[2], u4 = u[3]
    o[2] = param[1] + u[0] + u[2] + u[3];
    o[3] = param[1] + u[0] + u[1] - u[3];
    o[4] = param[1] + u[0] - u[1] - u[3];
    o[5] = o[1] * fabs(o[1]);
    o[6] = o[2] * fabs(o[2]);
    o[7] = o[3] * fabs(o[3]);
    o[8] = o[4] * fabs(o[4]);
    o[9] = o[5] + o[6] + o[7] + o[8];

    for(int i = 0; i < 13; i++)
    {
        c_state[i] = currentState[i];
    }

    // dot{X}
    dstate[0] = c_state[1];
    // dot{dot{X}}
    dstate[1] = 2.0 * param[4] * (c_state[9] * c_state[11] + c_state[10] * c_state[12]) * o[9] / o[0];
    // dot{Y}
    dstate[2] = c_state[3];
    // dot{dot{Y}}
    dstate[3] = -2.0 * param[4] * (c_state[9] * c_state[10] - c_state[11] * c_state[12]) * o[9] / o[0];
    // dot{Z}
    dstate[4] = c_state[5];
    // dot{dot{Z}}
    dstate[5] = (param[4]*(2.0 * c_state[9] * c_state[9] + 2.0 * c_state[12] * c_state[12] - 1.0) * o[9] / o[0]) - param[0];
    // dot{Gamma}
    dstate[6] = 0.5 * ( 2.0 * (param[7] - param[8]) * c_state[7] * c_state[8] + param[10] * param[4]* (o[7]-o[8]) / param[3]) / param[6];
    // dot{Beta}
    dstate[7] = -0.5 * ( 2.0 * (param[6] - param[8]) * c_state[6] * c_state[8] + param[10] * param[4]* (o[5]-o[6]) / param[3]) / param[7];
    // dot{alpha}
    dstate[8] = ((param[6] - param[7]) * c_state[6] * c_state[7] - param[5] * (o[7]+o[8]-o[5]-o[6])) / param[8];
    // dot{Quaternion_W}
    dstate[9] = -0.5 * c_state[10] * c_state[6] - 0.5 * c_state[11] * c_state[7] - 0.5 * c_state[12] * c_state[8];
    // dot{Quaternion_X}
    dstate[10] = 0.5 * c_state[9] * c_state[6] + 0.5 * c_state[11] * c_state[8] - 0.5 * c_state[12] * c_state[7];
    // dot{Quaternion_Y}
    dstate[11] = 0.5 * c_state[9] * c_state[7] + 0.5 * c_state[12] * c_state[6] - 0.5 * c_state[10] * c_state[8];
    // dot{Quaternion_Z}
    dstate[12] = 0.5 * c_state[9] * c_state[8] + 0.5 * c_state[10] * c_state[7] - 0.5 * c_state[11] * c_state[6];
}

__host__ __device__ double myStageCostFunction(double *u, double *st, double *reference, double *weightMatrix)
{
    double ret = 0.0;
    ret += weightMatrix[0] * (st[0] - reference[1]) * (st[0] - reference[1]); //q_11 * (x - ref{x})^2
    ret += weightMatrix[2] * (st[2] - reference[2]) * (st[2] - reference[2]); //q_33 * (y - ref{y})^2
    ret += weightMatrix[4] * (st[4] - reference[3]) * (st[4] - reference[3]); //q_55 * (z - ref{z})^2
    ret += weightMatrix[1] * st[1] * st[1] + weightMatrix[3] * st[3] * st[3] + weightMatrix[5] * st[5] * st[5]; // q_22 * dX^2 + q_44 * dY^2 + q_66 * dZ^2
    // q_77 * 
    for(int i = 6; i < 9; i++)
    {
        ret += weightMatrix[i] * st[i] * st[i];
    }

    ret += weightMatrix[9] * (u[0] - reference[0]) * (u[0] - reference[0]);
    ret += weightMatrix[10] * u[1] * u[1] + weightMatrix[11] * u[2] * u[2] + weightMatrix[12] * u[3] * u[3];

    ret = ret / 2;

    return ret;

} 