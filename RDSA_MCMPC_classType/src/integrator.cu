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