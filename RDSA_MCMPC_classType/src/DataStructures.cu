/* 
------2021.12.21 start making-------------
*/

#include "../include/DataStructures.cuh"

void init_structure(SampleInfo *info, int num, IndexParams *Idx)
{
    for(int i = 0; i < num; i++)
    {
        info[i].cost = 0.0;
        info[i].weight = 0.0;
        info[i].input = Idx->InputByHorizon;
        info[i].dev_dstate = Idx->dim_of_state;
        info[i].dev_state = Idx->dim_of_state;
        info[i].dev_input = Idx->dim_of_input;
    }
    printf("end of SamplInfo definition!!!! \n");
}

void init_structure(QHP *qhp, int num,  IndexParams *Idx)
{
    for(int i = 0; i < num; i++)
    {
        qhp[i].tensor_vector = Idx->HessianElements;
        qhp[i].column_vector = Idx->HessianElements;  
    }
}