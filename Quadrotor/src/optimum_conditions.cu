/*
    optimum condition
*/

#include "../include/optimum_conditions.cuh"

double input_saturation(double u, double Umax, double Umin)
{
    double ret_u = 0.0;
    if(u < Umin){
        ret_u = Umin + zeta;
    }else if(u > Umax){
        ret_u = Umax - zeta;
    }else{
        ret_u = u;
    }
    return ret_u;
}

void calc_Lx_Terminal_Cart_and_SinglePole(Tolerance *get, SystemControlVariable *SCV)
{
    get->lambda[0] = get->state[0] * SCV->weightMatrix[0];
    get->lambda[1] = cosf(get->state[1]/2) * sinf(get->state[1]/2) * SCV->weightMatrix[1];
    get->lambda[2] = get->state[2] * SCV->weightMatrix[2];
    get->lambda[3] = get->state[3] * SCV->weightMatrix[3];
}

void calc_Lx_Cart_and_SinglePole(Tolerance *get, SystemControlVariable *SCV)
{
    get->lambda[0] = get->state[0] * SCV->weightMatrix[0];
    get->lambda[1] = cosf(get->state[1]/2) * sinf(get->state[1]/2) * SCV->weightMatrix[1] / 2.0f;
    get->lambda[2] = get->state[2] * SCV->weightMatrix[2];
    get->lambda[3] = get->state[3] * SCV->weightMatrix[3];
}

// この関数で、最適性残山に関する計算が全て簡潔可能かも知れない。 2021.07.09
// calc_BackwardCollection_Cart_and_SinglePole  λの計算，（∂H/∂u）の計算（結果を　*Tolに返す関数）
void calc_BC_Cart_and_SinglePole(Tolerance *Tol, SystemControlVariable *SCV)
{
    double temp_Lambda[DIM_OF_STATES] = { };
    double temp_Lx[DIM_OF_STATES] = { };
    double temp_LBx[DIM_OF_STATES] = { };
    double temp_LFx[DIM_OF_STATES] = { };
#ifdef TerminalCost
    calc_Lx_Terminal_Cart_and_SinglePole(&Tol[HORIZON], SCV);
#else
    calc_Lx_Cart_and_SinglePole(&Tol[HORIZON], SCV);
#endif
    int p_index = 0;
    double d_sec = predictionInterval / HORIZON;
    for(int t = 1; t < HORIZON + 1; t++)
    {
        p_index = (HORIZON - t) + 1;
        get_Lx_Cart_and_SinglePole(temp_Lx, &Tol[HORIZON - t], SCV); //この関数の実体は未記述　2021.07.09
        // get_LBx_Cart_and_SinglePole(temp_LBx, &Tol[HORIZON - t], &Tol[p_index], SCV); //この関数の実体は未記述　2021.07.09 振子かつ状態制約無しの場合は不要
        // get_LFx_Cart_and_SinglePole(temp_LFx, &Tol[HORIZON - t], &Tol[p_index], SCV, d_sec); //この関数の実体は未記述　2021.07.09
        get_LFx_Using_M_Cart_and_SinglePole(temp_LFx, &Tol[HORIZON - t], &Tol[p_index], SCV, d_sec); //add 2021.8.26 write by NAK
        get_dHdu_Cart_and_SinglePole(&Tol[HORIZON-t], &Tol[p_index], SCV, d_sec);

        for(int i = 0; i < DIM_OF_STATES; i++)
        {
            temp_Lambda[i] = temp_Lx[i] + Rho * temp_LBx[i] + temp_LFx[i];
            // Tol[HORIZON - t].lambda[i] = Tol[p_index].lambda[i] + temp_Lambda[i] * d_sec; // lam_i = lam_{i+1} + dHdx_i * Dt
            Tol[HORIZON - t].lambda[i] = Tol[p_index].lambda[i] + temp_Lambda[i];
        }
    }
}

void calc_OC_for_Cart_and_SinglePole_hostF(double *Ans, double *U, SystemControlVariable *SCV, Tolerance *Tol)
{
    /*Tolerance *Tol;
    int T_size = HORIZON + 1;
    Tol = (Tolerance*)malloc(sizeof(Tolerance) * T_size);*/
    // Tolerance Tol[HORIZON+1];
    double costValue = 0.0;
    double stageCost = 0.0;
    double logBarrier = 0.0;
    double KKT_Error = 0.0;

    for(int index = 0; index < DIM_OF_STATES; index++)
    {
        // printf("State[%d] == %f\n", index, SCV->state[index]);
        Tol[0].state[index] = SCV->state[index];
    }

    double d_sec = predictionInterval / HORIZON;
    for(int t = 0; t < HORIZON; t++)
    {
        U[t] = input_saturation(U[t], SCV->constraints[1], SCV->constraints[0]);
        Tol[t].Input[0] = U[t]; //台車型１重直列倒立振子の場合
        Tol[t].dstate[0] = Tol[t].state[2]; //dx_{cur} = dx_{prev}
        Tol[t].dstate[1] = Tol[t].state[3]; //dtheta_{cur} = dtheta_{prev}
        Tol[t].dstate[2] = Cart_type_Pendulum_ddx(U[t], Tol[t].state[0], Tol[t].state[1], Tol[t].state[2], Tol[t].state[3], SCV);
        Tol[t].dstate[3] = Cart_type_Pendulum_ddtheta(U[t], Tol[t].state[0],  Tol[t].state[1], Tol[t].state[2], Tol[t].state[3], SCV);
        Tol[t+1].state[2] = Tol[t].state[2] + (d_sec * Tol[t].dstate[2]);
        Tol[t+1].state[3] = Tol[t].state[3] + (d_sec * Tol[t].dstate[3]);
        Tol[t+1].state[0] = Tol[t].state[0] + (d_sec * Tol[t].dstate[0]);
        Tol[t+1].state[1] = Tol[t].state[1] + (d_sec * Tol[t].dstate[1]);

        // printf("log[1] == %f, log[2] = %f\n", log(U[t] + SCV->constraints[1]), log(SCV->constraints[1] - U[t]));
        logBarrier = (SCV->constraints[1]- SCV->constraints[0]) * sRho - log(U[t] + SCV->constraints[1]) - log(SCV->constraints[1] - U[t]);
        /*stageCost = Tol[t+1].state[0] * Tol[t+1].state[0] * SCV->weightMatrix[0] + sinf(Tol[t+1].state[1] / 2) * sinf(Tol[t+1].state[1] / 2) * SCV->weightMatrix[1]
                    + Tol[t+1].state[2] * Tol[t+1].state[2] * SCV->weightMatrix[2] + Tol[t+1].state[3] * Tol[t+1].state[3] * SCV->weightMatrix[3]
                    + U[t] * U[t] * SCV->weightMatrix[4];*/
        stageCost = Tol[t].state[0] * Tol[t].state[0] * SCV->weightMatrix[0] + sinf(Tol[t+1].state[1] / 2) * sinf(Tol[t+1].state[1] / 2) * SCV->weightMatrix[1]
                    + Tol[t].state[2] * Tol[t].state[2] * SCV->weightMatrix[2] + Tol[t].state[3] * Tol[t].state[3] * SCV->weightMatrix[3]
                    + U[t] * U[t] * SCV->weightMatrix[4];

        stageCost = 0.5f * stageCost;
        // printf("StageCost := %f, LogBarrier := %f\n", stageCost, logBarrier);
        if(t == HORIZON -1)
        {
            stageCost += Tol[t+1].state[0] * Tol[t+1].state[0] * SCV->weightMatrix[0] + sinf(Tol[t+1].state[1] / 2) * sinf(Tol[t+1].state[1] / 2) * SCV->weightMatrix[1]
                        + Tol[t+1].state[2] * Tol[t+1].state[2] * SCV->weightMatrix[2] + Tol[t+1].state[3] * Tol[t+1].state[3] * SCV->weightMatrix[3];
        }
        
        // costValue += stageCost + Rho * logBarrier;
        costValue += stageCost;
        stageCost = 0.0;
        logBarrier = 0.0;

    }
    calc_BC_Cart_and_SinglePole(Tol, SCV);
    // KKT_Error = powf(Tol[0].dHdu[0], 2);
    KKT_Error = fabs(Tol[0].dHdu[0]);
    double candidate = 0.0;
    for(int i = 1; i < HORIZON; i++)
    {
        // candidate = powf(Tol[i].dHdu[0], 2);
        candidate = fabs(Tol[i].dHdu[0]);
        if(KKT_Error < candidate)
        {
            KKT_Error = candidate;
        }
    }

    Ans[0] = costValue;
    Ans[1] = KKT_Error;
    // free(Tol->lambda);
    // free(Tol);

}

void calc_Cost_Quadrotor(double *cost, double *U, SystemControlVariable *SCV)
{
    double costValue[2] = { };
    double stageCost = 0.0;
    double logBarrier = 0.0;
    double state[DIM_OF_STATES] = { };
    double dstate[DIM_OF_STATES] = { };

    for(int id_ST = 0; id_ST < DIM_OF_STATES; id_ST++)
    {
        state[id_ST] = SCV->state[id_ST];
    }

    int uIndex = 0;
    double d_time = predictionInterval / HORIZON;
    for(int hStep = 0;  hStep < HORIZON; hStep++)
    {
        uIndex = hStep * DIM_OF_INPUT;
        dynamics_ddot_Quadrotor(dstate, U[uIndex], U[uIndex + 1], U[uIndex + 2], U[uIndex + 3], state, SCV);
        for(int stIndex = 0; stIndex < DIM_OF_STATES; stIndex++)
        {
            state[stIndex] = state[stIndex] + ( dstate[stIndex] * d_time );
        }
        //
        logBarrier += -logf(U[uIndex+1]+SCV->constraints[3])-logf(U[uIndex+2]+SCV->constraints[3])-logf(U[uIndex+3]+SCV->constraints[3]);
        logBarrier += -logf(SCV->constraints[3]-U[uIndex+1])-logf(SCV->constraints[3]-U[uIndex+2])-logf(SCV->constraints[3]-U[uIndex+3]);
        logBarrier += -logf(state[6]+SCV->constraints[1])-logf(state[7]+SCV->constraints[1])-logf(state[8]+SCV->constraints[1]);
        logBarrier += -logf(SCV->constraints[1]-state[6])-logf(SCV->constraints[1]-state[7])-logf(SCV->constraints[1]-state[8]);
        logBarrier += -logf(SCV->constraints[5]-U[uIndex])-logf(U[uIndex]);
        logBarrier += sRho * ((SCV->constraints[3]-SCV->constraints[2])+(SCV->constraints[1]-SCV->constraints[0])+(SCV->constraints[5]-SCV->constraints[4]));

        stageCost = SCV->weightMatrix[0] * (state[0] - SCV->params[1]) * (state[0] - SCV->params[1])
                    + SCV->weightMatrix[2] * (state[2] - SCV->params[2]) * (state[2] - SCV->params[2])
                    + SCV->weightMatrix[4] * (state[4] - SCV->params[3]) * (state[4] - SCV->params[3])
                    + SCV->weightMatrix[1] * state[1] * state[1] + SCV->weightMatrix[3] * state[3] * state[3]
                    + SCV->weightMatrix[5] * state[5] * state[5] + SCV->weightMatrix[6] * state[6] * state[6]
                    + SCV->weightMatrix[7] * state[7] * state[7] + SCV->weightMatrix[8] * state[8] * state[8]
                    + SCV->weightMatrix[9] * (U[uIndex] - SCV->params[0]) * (U[uIndex] - SCV->params[0]) + SCV->weightMatrix[10] * U[uIndex+1] * U[uIndex+1]
                    + SCV->weightMatrix[11] * U[uIndex+2] * U[uIndex+2] + SCV->weightMatrix[12] * U[uIndex+3] * U[uIndex+3];
        
        stageCost = stageCost / 2;

        costValue[0] += stageCost;
        costValue[1] += stageCost + Rho * logBarrier;
        logBarrier = 0.0;
    }
    cost[0] = costValue[0];
    cost[1] = costValue[1];

}

