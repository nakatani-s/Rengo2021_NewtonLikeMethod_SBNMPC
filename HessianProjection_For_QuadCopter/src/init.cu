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
    int InputByHorizon = DIM_OF_INPUT * HORIZON;
    int size_of_Hessian = InputByHorizon * InputByHorizon;
    int size_of_HessianElements = (int)(size_of_Hessian * (size_of_Hessian + 1) / 2);
    gIdx->InputByHorizon = InputByHorizon;
    gIdx->dim_Hessian = size_of_Hessian;
    gIdx->sz_HessElements = size_of_HessianElements;
    gIdx->sz_pow_HessElements = size_of_ElmHessian * size_of_ElmHessian;

    int sampleSize = size_of_HessianElements;
    int counter = 0;
    while(!(sampleSize%THREAD_PER_BLOCKS==0) || counter < max_divisor )
    {
        sampleSize++;
        if(sampleSize%THREAD_PER_BLOCKS==0)
        {
            counter++;
        }
    }
    gIdx->sz_FittingSamples = sampleSize;
}

unsigned int countBlocks(unsigned int a, unsigned int b)
{
    unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}