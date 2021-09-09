/*
 dynamics.cu
*/
#include "../include/dynamics.cuh"

double Dynamics_InputSaturation(double U, double Lower, double Upper)
{
    double ret = U;
    if(ret < Lower)
    {
        ret = Lower + zeta;
    }
    if( ret > Upper)
    {
        ret = Upper - zeta;
    }
    return ret;
}

void Quadrotor_InputSaturation(double *U, SystemControlVariable *SCV)
{
    int uIndex;
    for(int i = 0; i < HORIZON; i++){
        uIndex = i * DIM_OF_INPUT;
        U[uIndex] = Dynamics_InputSaturation(U[uIndex], SCV->constraints[4], SCV->constraints[5]);
        for(int j = 1; j < DIM_OF_INPUT; j++)
        {
            U[uIndex+j] = Dynamics_InputSaturation(U[uIndex+j], SCV->constraints[2], SCV->constraints[3]);
        }
    }
}

void Quadrotor_Input_recalculation(double *U, SystemControlVariable *SCV)
{
    int uIndex;
    for(int i = 0; i < HORIZON; i++)
    {
        for(int j = 0; j < DIM_OF_INPUT; j++)
        {
            uIndex = i * DIM_OF_INPUT + j;
            if(uIndex % 4 == 0)
            {
                U[uIndex] = U[uIndex] * SCV->constraints[5];
            }else{
                U[uIndex] = U[uIndex];
            }
        }
    }
}


__host__ __device__ double Cart_type_Pendulum_ddx(double u, double x, double theta, double dx, double dtheta, SystemControlVariable *SCV)
{
    double a[10];
    /*a[0] = state[3] + powf(state[2], 2) * state[1];		//J+l^2*mp
	a[1] = u - dx * state[4]
			+ powf(dtheta, 2) * state[2] * state[1] * sinf(theta);//u-dx*myuc+dtheta^2*l*mp*sin
	a[2] = cosf(theta) * state[2] * state[1];						//cos*l*mp
	a[3] = dtheta * state[5] - state[6] * state[2] * state[1] * sinf(theta);//dtheta*myup-g*l*mp*sin
	a[4] = -(a[0] * a[1] + a[2] * a[3]);

	a[5] = powf(cosf(theta), 2) * powf(state[2], 2) * powf(state[1], 2);//cos^2*l^2*mp^2
	a[6] = state[0] + state[1];		//mc+mp
	a[7] = state[3] + powf(state[2], 2) * state[1];		//J+l^2*mp
	a[8] = a[5] - (a[6] * a[7]);*/

    a[0] = SCV->params[3] + powf(SCV->params[2], 2) * SCV->params[1]; // J + mplp^2
    a[1] = u - dx * SCV->params[4] + powf(dtheta, 2) * SCV->params[2] * SCV->params[1] * sinf(theta); // u - dx * myu_c *dtheta^2 * mp * lp * sin(theta)
    a[2] = cosf(theta) * SCV->params[2] * SCV->params[1]; // mp * lp * cos(theta)
    a[3] = dtheta * SCV->params[5] - SCV->params[6] * SCV->params[2] * SCV->params[1] * sinf(theta); //myu_p * dtheta - mp * g * lp * sin(theta)
    a[4] = -(a[0] * a[1] + a[2] * a[3]);

    a[5] = powf(cosf(theta), 2) * powf(SCV->params[2], 2) * powf(SCV->params[1], 2); //mp^2 * lp^2 * cos^2(theta)
    a[6] = SCV->params[0] + SCV->params[1]; // mp + mc
    a[7] = SCV->params[3] + powf(SCV->params[2],2) * SCV->params[1];
    a[8] = a[5] - (a[6] * a[7]);

	return a[4] / a[8];
}

__host__ __device__ double Cart_type_Pendulum_ddtheta(double u, double x,  double theta, double dx, double dtheta, SystemControlVariable *SCV)
{
    double a[10];
    /*a[0] = cosf(theta) * state[2] * state[1];		//cos*l*mp
	a[1] = u - dx * state[4]
			+ powf(dtheta, 2) * state[2] * state[1] * sinf(theta);//u-dx*myuc+dtheta^2*l*mp*sin
	a[2] = state[0] + state[1];		//mc+mp
	a[3] = dtheta * state[5] - state[6] * state[2] * state[1] * sinf(theta);//dtheta*myup-g*l*mp*sin
	a[4] = -(a[0] * a[1] + a[2] * a[3]);

	a[5] = state[3] * (state[0] + state[1]);		//J(mc+mp)
	a[6] = powf(state[2], 2) * state[1];		//l^2*mp
	a[7] = state[0] + state[1] - powf(cosf(theta), 2) * state[1];//mc+mp-cos^2*mp
	a[8] = a[5] + a[6] * a[7];*/

    a[0] = cosf(theta) * SCV->params[2] * SCV->params[1]; //mp * lp * cos(theta)
    a[1] = u - dx * SCV->params[4] + powf(dtheta, 2) * SCV->params[2] * SCV->params[1] * sinf(theta); //u - myu_c * dx + mp * lp * dtheta^2 *sin(theta)
    a[2] = SCV->params[0] + SCV->params[1]; //mc + mp
    a[3] = dtheta * SCV->params[5] - SCV->params[6] * SCV->params[2] * SCV->params[1] * sinf(theta); // myu_p * dtheta - mp * lp * sin(theta)
    a[4] = -(a[0] * a[1] + a[2] * a[3]);

    a[5] = SCV->params[3] * (SCV->params[0] + SCV->params[1]);
    a[6] = powf(SCV->params[2], 2) * SCV->params[1];
    a[7] = SCV->params[0] + SCV->params[1] - powf(cosf(theta), 2) * SCV->params[1];
    a[8] = a[5] + a[6] * a[7];

	return a[4] / a[8];
}

__host__ __device__ void dynamics_ddot_Quadrotor(double *dstate, double u1, double u2, double u3, double u4, double *c_state, SystemControlVariable *SCV)
{
    double gamm, beta, alpha;
    double mu = 0.0;
    gamm = c_state[6];
    beta = c_state[7];
    alpha = c_state[8];

    dstate[0] = c_state[1]; // dX
    dstate[2] = c_state[3]; // dY
    dstate[4] = c_state[5]; // dZ

    dstate[1] = u1 * ( cosf(gamm)*sinf(beta)*cosf(alpha) + sinf(gamm) * sinf(alpha) ); // ddX
    dstate[3] = u1 * ( cosf(gamm)*sinf(beta)*sinf(alpha) - sinf(gamm) * cosf(alpha) ); // ddY
    dstate[5] = u1 * cosf(gamm) * cosf(beta) - SCV->params[0]; // ddZ

    /*if(cosf(beta) == 0.0f)
    {
        dstate[6] = ( u2 * cosf(gamm) + u3 * sinf(gamm)) / zeta;
    }else{
        dstate[6] = (u2 * cosf(gamm) + u3 * sinf(gamm)) / cosf(beta);
    }*/
    mu = u2*cosf(gamm) + u3 * sinf(gamm);
    // dstate[6] = (u2 * cosf(gamm) + u3 * sinf(gamm)) / cosf(beta);
    dstate[6] = mu / cosf(beta);
    dstate[7] = -u2 * sinf(gamm) + u3 * cosf(gamm); // Beta
    dstate[8] = u2 * cosf(gamm) * tan(beta) + u3 * sinf(gamm) * tan(beta) + u4; //d_alpha
}

__host__ __device__ void dynamics_QuaternionBased_Quadrotor(double *dstate, double u1, double u2, double u3, double u4, double *c_state, SystemControlVariable *SCV)
{
    // double gamm, beta, alpha;
    // double mu = 0.0;
    double o[6] = { };
    o[0] = SCV->sparams[9] * SCV->sparams[3]; // M * u_max^2
    o[1] = u1 * fabs(u1); // St(1) = u(1) * |u(1)|
    o[2] = u2 * fabs(u2); // St(2) = u(2) * |u(2)|
    o[3] = u3 * fabs(u3); // St(3) = u(3) * |u(3)|
    o[4] = u4 * fabs(u4); // St(4) = u(4) * |u(4)|
    o[5] = o[1]+o[2]+o[3]+o[4]; // St(1)+St(2)+St(3)+St(4)

    /*gamm = c_state[6];
    beta = c_state[7];
    alpha = c_state[8];*/

    dstate[0] = c_state[1]; // dX
    dstate[2] = c_state[3]; // dY
    dstate[4] = c_state[5]; // dZ

    dstate[1] = 2.0 * SCV->sparams[4]*(c_state[9]*c_state[11] + c_state[10]*c_state[12])*o[5] / o[0]; //ddX
    dstate[3] = -2.0 * SCV->sparams[4] * (c_state[9]*c_state[10] - c_state[11]*c_state[12])*o[5] / o[0]; //ddY
    dstate[5] = (SCV->sparams[4]*(2.0 * c_state[9] * c_state[9] + 2.0 * c_state[12] * c_state[12] - 1.0) * o[5] / o[0]) - SCV->sparams[0]; // ddZ

    // 0.5*(2.0*(Iyy-Izz)*Beta*Alpha+distL*T_max*(St(3)-St(4))/(Umpow))/Ixx
    // -0.5*(2.0*(Ixx-Izz)*Gamma*Alpha+distL*T_max*(St(1)-St(2))/(Umpow))/Iyy
    // ((Ixx-Iyy)*Gamma*Beta-Tau_max*(St(3)+St(4)-St(1)-St(2)))/Izz
    dstate[6] = 0.5 * ( 2.0 * (SCV->sparams[7] - SCV->sparams[8]) * c_state[7] * c_state[8] + SCV->sparams[10] * SCV->sparams[4]* (o[3]-o[4]) / SCV->sparams[3]) / SCV->sparams[6]; // Gamma
    dstate[7] = -0.5 * ( 2.0 * (SCV->sparams[6] - SCV->sparams[8]) * c_state[6] * c_state[8] + SCV->sparams[10] * SCV->sparams[4]* (o[1]-o[2]) / SCV->sparams[3]) / SCV->sparams[7]; // Beta
    dstate[8] = ((SCV->sparams[6]-SCV->sparams[7])*c_state[6]*c_state[7]-SCV->sparams[4]*(o[3]+o[4]-o[1]-o[2]))/ SCV->sparams[8]; // alpha

    // -0.5*QuaX*Gamma-0.5*QuaY*Beta-0.5*QuaZ*Alpha;
    // 0.5*QuaW*Gamma+0.5*QuaY*Alpha-0.5*QuaZ*Beta;
    // 0.5*QuaW*Beta+0.5*QuaZ*Gamma-0.5*QuaX*Alpha;
    // 0.5*QuaW*Alpha+0.5*QuaX*Beta-0.5*QuaY*Gamma;
    dstate[9] = -0.5 * c_state[10] * c_state[6] - 0.5 * c_state[11] * c_state[7] - 0.5 * c_state[12] * c_state[8]; // Quaternion_W
    dstate[10] = 0.5 * c_state[9] * c_state[6] + 0.5 * c_state[11] * c_state[8] - 0.5 * c_state[12] * c_state[7]; // Quaternion_X
    dstate[11] = 0.5 * c_state[9] * c_state[7] + 0.5 * c_state[12] * c_state[6] - 0.5 * c_state[10] * c_state[9]; // Quaternion_Y
    dstate[12] = 0.5 * c_state[9] * c_state[8] + 0.5 * c_state[10] * c_state[7] - 0.5 * c_state[11] * c_state[6]; // Quaternion_Z
}

__host__ __device__ void get_Lx_Cart_and_SinglePole(double *Lx, Tolerance *prev, SystemControlVariable *SCV)
{
    // double temp_Lx[DIM_OF_STATES] = { };
    Lx[0] = SCV->weightMatrix[0] * prev->state[0];
    Lx[1] = SCV->weightMatrix[1] * cosf(prev->state[1] / 2) * sinf(prev->state[1] / 2) * 0.5;
    Lx[2] = SCV->weightMatrix[2] * prev->state[2];
    Lx[3] = SCV->weightMatrix[3] * prev->state[3];
}

__host__ __device__ void get_LFx_Cart_and_SinglePole(double *LFx, Tolerance *current, Tolerance *later, SystemControlVariable *SCV, double t_delta)
{
    double a[36] = { };
    double /*x,*/ th, dx, dtheta;
    // x = current->state[0];
    th = current->state[1];
    dx = current->state[2];
    dtheta = current->state[3];

    a[0] = - later->lambda[0]; //lambda^T * Fx(:,1) := LFx[0]

    a[1] = -powf(SCV->params[1] * SCV->params[2] * cosf(th), 2) + powf(SCV->params[1] * SCV->params[2], 2)
            + SCV->params[0] * SCV->params[1] * powf(SCV->params[2], 2) + SCV->params[3] * (SCV->params[0] + SCV->params[1]); // -(Mp*lp*cos(th))^2 + (Mp*lp)^2 + Mc*Mp*lp^2 + Jp * (Mp + Mc)
    a[2] = (SCV->params[1] * powf(SCV->params[2], 2) + SCV->params[3]) * SCV->params[1] * powf(dtheta, 2) * SCV->params[2] * cosf(th); // (Jp + Mp*lp^2) * Mp * lp * dth^2 * cos(th)
    a[3] = powf(SCV->params[1] * SCV->params[2] * cosf(th), 2) * SCV->params[6]; //(Mp*lp*cos(th))^2 * g
    a[4] = (SCV->params[5] * dtheta - SCV->params[1] * SCV->params[6] * SCV->params[2] * sinf(th)); // (mup * dtheta - Mp * lp * g * sin(th) )
    a[5] = SCV->params[1] * SCV->params[2] * sinf(th); //Mp * lp * sin(th)
    a[6] = a[4] * a[5]; //Mp * lp * sin(th) *(mup * dtheta - Mp * lp * g * sin(th) )
    a[7] = 2.0f * powf(SCV->params[1] * SCV->params[2], 3) * powf(cosf(th), 2) * sinf(th); //2 * Mp^3 * lp^3 * cos^2(th) * sin(th)
    a[8] = 2.0f * powf(SCV->params[1] * SCV->params[2], 2) * cosf(th) * sinf(th); //2 * Mp^2 * lp^2 * cos(th) * sin(th)
    a[9] = SCV->params[3] + SCV->params[1] * powf(SCV->params[2], 2); // Jp + Mp * lp^2
    a[10] = SCV->params[1] * SCV->params[2] * powf(dtheta, 2) * sinf(th) + current->Input[0] - SCV->params[4] * dx; // [Mp * lp * dtheta^2 * sinf(th) + U - muc * dx]
    a[11] = SCV->params[1] * SCV->params[2]; //Mp * lp
    // a[12] = 2.0f * SCV->params[1] * powf( SCV->params[2], 2) * dtheta * sinf(th) + SCV->params[5] * cosf(th) + 2.0f * SCV->params[3] * sinf(th); //2 * Mp * lp^2 *dtheta * sin(th) + mup * cos(th) + 2 * Jp * dtheta * sin(th)
    a[12] = 2.0f * SCV->params[1] * powf( SCV->params[2], 2) * dtheta * sinf(th) + SCV->params[5] * cosf(th) + 2.0f * SCV->params[3] * dtheta * sinf(th); //2 * Mp * lp^2 *dtheta * sin(th) + mup * cos(th) + 2 * Jp * dtheta * sin(th)
    
    a[13] = a[2] / a[1]; //[Fx;32]_1
    a[14] = a[3] / a[1]; //[Fx;32]_2
    a[15] = a[6] / a[1]; //[Fx;32]_3
    a[16] = a[1] * a[1]; //a[1]^2
    a[17] = (a[7] * a[4]) / a[16]; //[Fx;32]_4
    a[18] = (a[8] * a[9] * a[10]) / a[16]; //[Fx;32]_5
    a[19] = (a[13] - a[14] - a[15] - a[17] - a[18]) * t_delta; // Fx;32
    
    a[20] = -(SCV->params[4] * a[9]) / a[1]; //[Fx;33] = -myuc * (Jp + Mp * lp^2) / a[1]
    a[21] = a[20] * t_delta - 1.0; // Fx;33

    a[22] = SCV->params[1] * SCV->params[2] * a[12] / a[1]; //[Fx;34] = Mp * lp * (2 * Mp * lp^2 * dtheta * sin(th) + myup * cos(th) + 2 * Jp * dtheta * sinf(th) ) / a[1]
    a[23] = a[22] * t_delta; //Fx;34

    a[24] = (a[5] * a[10]) / a[1]; //[Fx;42]_1
    a[25] =  powf(SCV->params[1] * SCV->params[2] * cosf(th), 2); // (Mp*lp*cos(th))^2
    // a[26] = a[22] * powf(dtheta, 2) / a[1]; //[Fx;42]_2
    a[26] = a[25] * powf(dtheta,2) / a[1];
    a[27] = a[11] * SCV->params[6] * cosf(th) * (SCV->params[0] + SCV->params[1]) / a[1]; //[Fx;42]_3
    a[28] = (a[7] * a[10]) / a[16]; //[Fx;42]_4
    a[29] = (a[8] * (SCV->params[0] + SCV->params[1]) * a[4]) / a[16]; //[Fx;42]_5
    a[30] = (a[24] - a[26] + a[27] + a[28] + a[29]) * t_delta; // Fx;42 

    a[31] = SCV->params[5] * SCV->params[1] * SCV->params[2] * cosf(th) / a[1]; //[Fx;43]
    a[32] = a[31] * t_delta; //Fx;43

    a[33] = -SCV->params[5] * (SCV->params[0] + SCV->params[1]) / a[1]; //[Fx;44]_1
    a[34] = -a[8] * dtheta / a[1]; //[Fx;44]_2
    // a[35] = (a[29] + a[30]) * t_delta - 1.0f; //Fx;44
    a[35] = (a[33]+a[34]) * t_delta - 1.0;

    // 行列Fxの面倒臭い要素の計算まで”一応終了”<--- 2021.07.12

    LFx[0] = a[0];
    LFx[1] = -later->lambda[1] + a[19] * later->lambda[2] + a[30] * later->lambda[3];
    LFx[2] = (t_delta * later->lambda[0]) + (a[21] * later->lambda[2]) + (a[32] * later->lambda[3]);
    LFx[3] = (t_delta * later->lambda[1]) + (a[23] * later->lambda[2]) + (a[35] * later->lambda[3]); 
}

// 2021.8.25 add 
__host__ __device__ void get_LFx_Using_M_Cart_and_SinglePole(double *LFx, Tolerance *current, Tolerance *later, SystemControlVariable *SCV, double t_delta)
{
    double o[35] = { };
    double th, dx, dtheta;
    th = current->state[1];
    dx = current->state[2];
    dtheta = current->state[3];

    o[0] = -powf(SCV->params[1]*SCV->params[2]*cosf(th),2); //-Mp^2lp^2cos^2(th)
    o[1] = powf(SCV->params[1]*SCV->params[2], 2); //Mp^2*lp^2
    o[2] = SCV->params[0]*SCV->params[1]*powf(SCV->params[2],2); //Mc*Mp*lp^2
    o[3] = SCV->params[3]*(SCV->params[0]+SCV->params[1]); // Jp*(Mc + Mp)

    o[4] = o[0]+o[1]+o[2]+o[3]; //Denom

    o[5] = SCV->params[3]+SCV->params[1]*powf(SCV->params[2], 2); //(Jp + Mp * lp^2)
    o[6] = SCV->params[0] + SCV->params[1]; //(Mp + Mc)
    o[7] = SCV->params[1]*SCV->params[2]*cosf(th); // (Mp * lp * cos(th))
    o[8] = SCV->params[5]*dtheta-SCV->params[1]*SCV->params[6]*SCV->params[2]*sinf(th); //(mup * dtheta - Mp * g * lp * sin(th) )
    o[9] = SCV->params[1]*SCV->params[2]*powf(dtheta,2)*sinf(th) + current->Input[0] - SCV->params[4]*dx; //(Mp*lp*dtheta^2 + U - muc*dx)
    o[10] = SCV->params[1]*SCV->params[2]*sinf(th); //Mp * lp * sin(th)
    o[11] = o[7] * powf(dtheta,2); //Mp * lp * dtheta^2 * cos(th)
    o[12] = o[7] * SCV->params[6]; //Mp * g * lp * cos(th)
    
    o[13] = o[6]*o[8]*t_delta / o[4]; // A
    o[14] = o[7]*o[9]*t_delta / o[4]; // B
    o[15] = o[5]*o[9]*t_delta / o[4]; // C
    o[16] = o[7]*o[8]*t_delta / o[4]; // D
    
    o[17] = o[10] * (o[13]+o[14]); // Mp * lp * sin(th) * [A + B]
    o[18] = o[10] * (o[15]+o[16]); // Mp * lp * sin(th) * [C + D]

    o[19] = o[5] * (o[17] - o[11]*t_delta) / o[4]; //[F_{x;32}]_1
    o[20] = o[7] * (o[18] + o[12]*t_delta) / o[4]; //[F_{x;32}]_2

    o[21] = o[19] - o[20]; //F_{x;32}

    o[22] = SCV->params[4]*o[5]*t_delta / o[4];
    
    o[23] = o[22] - 1.0; //F_{x;33}

    o[24] = SCV->params[5]*o[7]*t_delta; //mu_p * Mp * lp * cos(th) * delta_t
    o[25] = 2.0f*dtheta*o[10]*o[5]*t_delta; //2 * Mp * lp * dth * sin(th) * (Jp + Mp * lp) * t_delta

    o[26] = (o[24] + o[25]) / o[4]; //F_{x;34}

    o[27] = o[6]*(o[17] + o[12]*t_delta); // [F_{x;42}]_1
    o[28] = o[7]*(o[17] - o[11]*t_delta); // [F_{x;42}]_2

    o[29] = o[27] + o[28]; // F_{x;42}

    o[30] = SCV->params[4]*o[7]*t_delta / o[4]; // F_{x;43}
    
    o[31] = SCV->params[5]*o[6]*t_delta; //mu_p * (Mc + Mp) * delta_t
    o[32] = 2.0f*o[1]*powf(dtheta,2)*cosf(th)*sinf(th)*t_delta; //2 * Mp^2 * lp^2 * dtheta^2 * cos(th) * sin(th) * delta_t
    o[33] = (o[31] - o[32]) / o[4];
    
    o[34] = o[33] - 1.0; // F_{x;44}
    
    LFx[0] = -1.0f * later->lambda[0];
    LFx[1] = (-1.0f * later->lambda[1]) + (o[21] * later->lambda[2]) + (o[29] * later->lambda[3]);
    LFx[2] = (t_delta * later->lambda[0]) + (o[23] * later->lambda[2]) + (o[30] * later->lambda[3]);
    LFx[3] = (t_delta * later->lambda[1]) + (o[26] * later->lambda[2]) + (o[34] * later->lambda[3]);
}

__host__ __device__ void get_dHdu_Cart_and_SinglePole(Tolerance *current, Tolerance *later, SystemControlVariable *SCV, double t_delta)
{
    double temp_Lu[DIM_OF_INPUT] = { };
    double temp_LBu[DIM_OF_INPUT] = { };
    double temp_Fu[DIM_OF_STATES] = { };
    double temp_LamFu[DIM_OF_INPUT] = { };

    double o[10] = { };
    temp_Lu[0] = SCV->weightMatrix[4] * current->Input[0];
    
    o[0] = powf(current->Input[0], 2) - powf(SCV->constraints[1], 2); // (U^2 - U_max^2)
    o[1] = 2.0f * current->Input[0]; // 2 * U
    o[2] = o[1] / o[0]; //LBu = 2U / (U^2 - U_max^2)

    o[3] = SCV->params[3] + SCV->params[1] * powf(SCV->params[2], 2); //Jp + Mp * lp^2
    o[4] = -SCV->params[1] * SCV->params[2] * cosf(current->state[1]); // -Mp * lp * cos(th)
    
    o[5] = -powf(SCV->params[1] * SCV->params[2] * cosf(current->state[1]), 2); //- Mp^2 * lp^2 * cos(th)^2
    o[6] = powf(SCV->params[1] * SCV->params[2], 2); // Mp^2 * lp^2
    o[7] = SCV->params[0] * SCV->params[1] * powf(SCV->params[2], 2); // Mc * Mp * lp^2
    o[8] = SCV->params[3] * (SCV->params[0] + SCV->params[1]); // Jp * (Mp + Mc)
    o[9] = o[5] + o[6] + o[7] + o[8];

    temp_LBu[0] = -o[2];

    temp_Fu[2] = (o[3] * t_delta) / o[9];
    temp_Fu[3] = (o[4] * t_delta) / o[9];

    for(int i = 0; i < DIM_OF_INPUT; i++)
    {
        for(int k = 0; k < DIM_OF_STATES; k++)
        {
            temp_LamFu[i] += later->lambda[k] * temp_Fu[k];
        }
    }

    for(int i = 0; i < DIM_OF_INPUT; i++)
    {
        current->dHdu[i] = temp_Lu[i] + Rho * temp_LBu[i] + temp_LamFu[i];
    }
}

void get_curent_diff_state_Cart_and_SinglePole(double *diffState, double *state, double input, SystemControlVariable *SCV)
{
    diffState[0] = state[2]; //dx
    diffState[1] = state[3]; //dtheta
    diffState[2] = Cart_type_Pendulum_ddx(input, state[0], state[1], state[2], state[3], SCV);
    diffState[3] = Cart_type_Pendulum_ddtheta(input, state[0], state[1], state[2], state[3], SCV);
}

void Eular_integrator(double *yp_vector, double t_delta, double *diffState)
{
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        yp_vector[i] = diffState[i] * t_delta;
    }
}

void Runge_Kutta45_for_SecondaryOderSystem(SystemControlVariable *SCV, double input, double t_delta)
{
    double state[DIM_OF_STATES], diff_state[DIM_OF_STATES], yp_1[DIM_OF_STATES], next_state[DIM_OF_STATES];
    // double params[DIM_OF_PARAMETERS] = { };
    // copyStateFromDataStructure(state, SCV);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        state[i] = SCV->state[i];
    }
    /*for(int i = 0; i < DIM_OF_PARAMETERS; i++){
        params[i] = SCV->params[i];
    }*/
    get_curent_diff_state_Cart_and_SinglePole( diff_state, state, input, SCV);
    Eular_integrator(yp_1, t_delta, diff_state);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_1[i] / 2;
    }

    double yp_2[DIM_OF_STATES] = { };
    get_curent_diff_state_Cart_and_SinglePole( diff_state, next_state, input, SCV);
    Eular_integrator(yp_2, t_delta, diff_state);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_2[i] / 2;
    }

    double yp_3[DIM_OF_STATES] = { };
    get_curent_diff_state_Cart_and_SinglePole( diff_state, next_state, input, SCV);
    Eular_integrator(yp_3, t_delta, diff_state);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_3[i];
    }

    double yp_4[DIM_OF_STATES] = { };
    get_curent_diff_state_Cart_and_SinglePole( diff_state, next_state, input, SCV);
    Eular_integrator(yp_4, t_delta, diff_state);

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        SCV->state[i] = state[i] + (yp_1[i] + 2 * yp_2[i] + 2 * yp_3[i] + yp_4[i]) / 6.0;
    }
}

void Runge_Kutta45_Quadrotor(SystemControlVariable *SCV, double *input, double t_delta)
{
    double state[DIM_OF_STATES], diff_state[DIM_OF_STATES], yp_1[DIM_OF_STATES], next_state[DIM_OF_STATES];

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        state[i] = SCV->state[i];
    }

    dynamics_ddot_Quadrotor(diff_state, input[0], input[1], input[2], input[3], state, SCV);
    Eular_integrator(yp_1, t_delta, diff_state);

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_1[i] / 2;
    }

    double yp_2[DIM_OF_STATES] = { };
    dynamics_ddot_Quadrotor(diff_state, input[0], input[1], input[2], input[3], next_state, SCV);
    Eular_integrator(yp_2, t_delta, diff_state);

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_2[i] / 2;
    }

    double yp_3[DIM_OF_STATES] = { };
    dynamics_ddot_Quadrotor(diff_state, input[0], input[1], input[2], input[3], next_state, SCV);
    Eular_integrator(yp_3, t_delta, diff_state);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_3[i];
    }

    double yp_4[DIM_OF_STATES] = { };
    dynamics_ddot_Quadrotor(diff_state, input[0], input[1], input[2], input[3], next_state, SCV);
    Eular_integrator(yp_4, t_delta, diff_state);

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        SCV->state[i] = state[i] + (yp_1[i] + 2 * yp_2[i] + 2 * yp_3[i] + yp_4[i]) / 6.0;
    }

}

void Runge_Kutta45_QuaternionBased_Quadrotor(SystemControlVariable *SCV, double *input, double t_delta)
{
    double state[DIM_OF_STATES], diff_state[DIM_OF_STATES], yp_1[DIM_OF_STATES], next_state[DIM_OF_STATES];

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        state[i] = SCV->state[i];
    }

    dynamics_ddot_Quadrotor(diff_state, input[0], input[1], input[2], input[3], state, SCV);
    Eular_integrator(yp_1, t_delta, diff_state);

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_1[i] / 2;
    }

    double yp_2[DIM_OF_STATES] = { };
    dynamics_QuaternionBased_Quadrotor(diff_state, input[0], input[1], input[2], input[3], next_state, SCV);
    Eular_integrator(yp_2, t_delta, diff_state);

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_2[i] / 2;
    }

    double yp_3[DIM_OF_STATES] = { };
    dynamics_QuaternionBased_Quadrotor(diff_state, input[0], input[1], input[2], input[3], next_state, SCV);
    Eular_integrator(yp_3, t_delta, diff_state);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_3[i];
    }

    double yp_4[DIM_OF_STATES] = { };
    dynamics_QuaternionBased_Quadrotor(diff_state, input[0], input[1], input[2], input[3], next_state, SCV);
    Eular_integrator(yp_4, t_delta, diff_state);

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        SCV->state[i] = state[i] + (yp_1[i] + 2 * yp_2[i] + 2 * yp_3[i] + yp_4[i]) / 6.0;
    }
}