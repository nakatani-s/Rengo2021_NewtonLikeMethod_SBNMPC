/*
This file includes main function for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
＊高速化（正規方程式の係数行列の計算はcublasにお任せ）
＊gradient相当の係数はRDSAを使用
＊シミュレーションモデルは、クアッドコプター（外乱ありからの姿勢回復）
*/ 

#include "include/rdsa_mcmpc.cuh"

int main(int argc, char **argv)
{
    rdsa_mcmpc myMPC(NoCooling);
    // state ------------------------------->> X,   dX,  Y,   dY,  Z,   dZ,  dγ, dβ, dα, q1,  q2,, q3,  q4
    double state[OCP::DIM_OF_SYSTEM_STATE] = { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    double thrust_max = 230.0 * 230.0;
    double param[OCP::DIM_OF_SYSTEM_PARAMS] = {9.806650, 150.0, 230.0,thrust_max, 5.0, 4.0e-6, 0.0085, 0.008, 0.0165, 1.0, 0.5};
    double u[OCP::DIM_OF_INPUT] = {9.8066, 0.0, 0.0, 0.0};
    double constraint[OCP::DIM_OF_CONSTRAINTS] = {-0.2, 0.2, -20.0, 20.0, 0.0, 25.0};
    double w_matrix[OCP::DIM_OF_WEIGHT_MATRIX] = { };
    w_matrix[0] = 5.0; /*  <<-- Q11 */ 
    w_matrix[1] = 0.05; /* <<-- Q22 */
    w_matrix[2] = 5.0; /*  <<-- Q33 */
    w_matrix[3] = 0.05; /* <<-- Q44 */
    w_matrix[4] = 10.0; /* <<-- Q55 */
    w_matrix[5] = 0.1; /*  <<-- Q66 */
    w_matrix[6] = 0.05; /* <<-- Q77 */
    w_matrix[7] = 0.05; /* <<-- Q88 */
    w_matrix[8] = 0.05; /* <<-- Q99 */
    w_matrix[9] = 50.0; /* <<-- Q1010 */
    w_matrix[10] = 50.0; /* <<-- Q1111 */
    w_matrix[11] = 50.0; /* <<-- Q1212 */

    w_matrix[12] = 0.1; /* <<--R11 */
    w_matrix[13] = 0.1; /* <<--R22 */
    w_matrix[14] = 0.1; /* <<--R33 */
    w_matrix[15] = 0.1; /* <<--R44 */

    myMPC.set(state, setState);
    myMPC.set(param, setParameter);
    myMPC.set(u, setInput);
    myMPC.set(u, setReference);
    myMPC.set(constraint, setConstraint);
    myMPC.set(w_matrix, setWeightMatrix);

    for(int t = 0; t < OCP::SIM_STEPS; t++)
    {
        myMPC.execute_rdsa_mcmpc(u);
        myMPC.do_forward_simulation(state, u, RUNGE_KUTTA_45);
        myMPC.set(state, setState);
        myMPC.write_data_to_file(u);
    }
}