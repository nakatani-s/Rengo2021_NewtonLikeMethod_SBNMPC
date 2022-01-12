/* 
------2021.12.21 start making-------------
*/

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <curand_kernel.h>
#include "myController.cuh"

#ifndef DATASTRUCTURE_CUH
#define DATASTRUCTURE_CUH

class Managed
{
public:
    void *operator new(size_t len);
    void *operator new[](size_t len);
    void operator delete(void *ptr);
};

class DynamicalArray : public Managed
{
    int length;
    double *data;
public:
    // Default Constructor
    DynamicalArray();
    // Copy constructor
    DynamicalArray(const DynamicalArray& x);
    // Assignment operator
    DynamicalArray& operator=(const int size);
    // destructor
    ~DynamicalArray();
    // Access operator (from host or device)
    __host__ __device__ double& operator[](int pos);
    __host__ __device__ double* d_pointer();
private:
    void _realloc_data(int len);
};

struct SampleInfo : public Managed
{
    double cost;
    double weight;
    DynamicalArray input;
    DynamicalArray dev_state;
    DynamicalArray dev_input;
    DynamicalArray dev_dstate;
};

struct QHP : public Managed
{
    DynamicalArray tensor_vector;
    DynamicalArray column_vector;
};

// DynamicalArrayクラスのデストラクタを呼ぶのに使用する　（安全に終了するために必要）
template <typename TYPE> void launch_by_value(TYPE data){
    int milli_second = 10;
    usleep(milli_second); 
}

typedef struct{
    int horizon;
    int dim_of_input;
    int dim_of_state;
    int sample_size;
    int elite_sample_size;
    unsigned int InputByHorizon;
    unsigned int HessianSize;
    unsigned int HessianElements;
    unsigned int PowHessianElements;
    unsigned int FittingSampleSize;
    double control_cycle;
    double predict_interval; 
    double zeta;
    double sRho;
    double micro;
}IndexParams;

void init_structure(SampleInfo *info, int num, IndexParams *Idx);
void init_structure(QHP *qhp, int num, IndexParams *Idx);

enum valueType{
    setState, setInput, setParameter, setConstraint, setWeightMatrix, setReference
};
enum CoolingMethod{
    Geometric, Hyperbolic, NoCooling
};
enum IntegralMethod{
    EULAR, RUNGE_KUTTA_45
};

#endif