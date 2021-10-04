#include <stdio.h>
#include "params.cuh"
#include "DataStructure.cuh"
// #include "dynamics.cuh"

typedef struct{
    char name[30];
    char inputfile[35];
    int dimSize;
}dataName;

void get_timeParam(int *tparam,int month, int day, int hour, int min, int step);
void write_Matrix_Information(double *data, dataName *d_name, int *timeparam);
void write_Vector_Information(double *data, dataName *d_name);

void resd_InitSolution_Input(double *input, dataName *d_name);