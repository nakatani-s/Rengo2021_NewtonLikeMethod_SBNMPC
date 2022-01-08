/*
    積分近似のメソッドを定義
    *
    *
*/
#include "../include/integrator.cuh"

__host__ __device__ void transition_Eular(double *st, double *dst, double delta, int dimState)
{
    for(int st_id = 0; st_id < dimState; st_id++)
    {
        st[st_id] = st[st_id] + (delta * dst[st_id]);
    }
}

__host__ __device__ void runge_kutta_45(double *st, int state_dim, double *input, double *prm, double t_delta)
{
    double *diff_state, *yp_1, *next_state;
    double *yp_2, *yp_3, *yp_4;
    // hstate = (double *)malloc(sizeof(double) * Idx->dim_of_state);
    diff_state = (double *)malloc(sizeof(double) * state_dim);
    yp_1 = (double *)malloc(sizeof(double) * state_dim);
    yp_2 = (double *)malloc(sizeof(double) * state_dim);
    yp_3 = (double *)malloc(sizeof(double) * state_dim);
    yp_4 = (double *)malloc(sizeof(double) * state_dim);
    next_state = (double *)malloc(sizeof(double) * state_dim);
    
    for(int i=0; i < state_dim; i++){
        yp_1[i] = 0.0;
        yp_2[i] = 0.0;
        yp_3[i] = 0.0;
        yp_4[i] = 0.0;
    }

    myDynamicModel(diff_state, input, st, prm);
    transition_Eular(yp_1, diff_state, t_delta, state_dim);
    for(int i = 0; i < state_dim; i++)
    {
        next_state[i] = st[i] + yp_1[i] / 2;
    }

    myDynamicModel(diff_state, input, next_state, prm);
    transition_Eular(yp_2, diff_state, t_delta, state_dim);
    for(int i = 0; i < state_dim; i++)
    {
        next_state[i] = st[i] + yp_2[i] / 2;
    }

    myDynamicModel(diff_state, input, next_state, prm);
    transition_Eular(yp_3, diff_state, t_delta, state_dim);
    for(int i = 0; i < state_dim; i++)
    {
        next_state[i] = st[i] + yp_3[i];
    }

    myDynamicModel(diff_state, input, next_state, prm);
    transition_Eular(yp_4, diff_state, t_delta, state_dim);

    for(int i = 0; i < state_dim; i++)
    {
        st[i] = st[i] +(yp_1[i] + 2 * yp_2[i] + 2 * yp_3[i] + yp_4[i]) / 6.0;
    }

    free(diff_state);
    free(yp_1);
    free(yp_2);
    free(yp_3);
    free(yp_4);
    free(next_state);
}