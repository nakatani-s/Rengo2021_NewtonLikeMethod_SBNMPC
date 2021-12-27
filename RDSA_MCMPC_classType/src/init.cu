/*
This file includes all param initializing function for RDSA_NLMPC for Quaterion based Quad-Roter
author: Nakatani-s
Date 2021.12.21
For SICE journal
Random Direction Stochastic Approximation + Sample based Newton-like Method
*/

#include "../include/init.cuh"

void set_IdxParams(IndexParams *gIdx)
{
    // もし、パラメータファイル化するならここをいじれば良さそう
    gIdx->horizon = CONTROLLER::HORIZON;
    gIdx->dim_of_input = OCP::DIM_OF_INPUT;
    gIdx->dim_of_state = OCP::DIM_OF_SYSTEM_STATE;
    gIdx->sample_size = CONTROLLER::NUM_OF_SAMPLES;
    int IBH = OCP::DIM_OF_INPUT * CONTROLLER::HORIZON;
    int HessSize = IBH * IBH;
    int HessElementsNum = (int)(HessSize * (HessSize + 1) / 2);
    gIdx->InputByHorizon = IBH;
    gIdx->HessianSize = HessSize;
    gIdx->HessianElements = HessElementsNum;
    gIdx->PowHessianElements = HessElementsNum * HessElementsNum;

    int sampleSize = HessElementsNum;
    int counter = 0;
    int tpb = CONTROLLER::THREAD_PER_BLOCKS;
    while(!(sampleSize%tpb==0) || counter < CONTROLLER::MAX_DIVISOR )
    {
        sampleSize++;
        if(sampleSize%tpb==0)
        {
            counter++;
        }
    }
    gIdx->FittingSampleSize = sampleSize;
    gIdx->control_cycle = CONTROLLER::CONTROL_CYCLE;
    gIdx->predict_interval = CONTROLLER::PREDICTION_INTERVAL;
    gIdx->zeta = CONTROLLER::zeta;
    gIdx->sRho = CONTROLLER::sRho;
}