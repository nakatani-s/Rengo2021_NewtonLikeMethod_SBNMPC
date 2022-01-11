/* 
------2021.12.21 start making-------------
*/

#include "../include/DataStructures.cuh"

void init_structure(SampleInfo *info, int num, int dim)
{
    for(int i = 0; i < num; i++)
    {
        info[i].cost = 0.0;
        info[i].weight = 0.0;
        info[i].input = dim;
    }
}

void init_structure(QHP *qhp, int num, int dim)
{
    for(int i = 0; i < num; i++)
    {
        qhp[i].tensor_vector = dim;
        qhp[i].column_vector = dim;  
    }
}