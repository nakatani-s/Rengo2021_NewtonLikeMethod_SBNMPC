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
private:
    void _realloc_data(int len);
};

struct SampleInfo : public Managed
{
    double cost;
    double weight;
    DynamicalArray input;
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

void init_structure(SampleInfo *info, int num, int dim);
void init_structure(QHP *qhp, int num, int dim);
/*class SampleInfo
{
public:
    SampleInfo(); //コンストラクタ
    ~SampleInfo(); //デストラクタ
    double cost;
    double weight;
    double *inputSeq;
};*/
// 以下の構造体は、cudaのユニファイドメモリで管理
/*class SystemControlVariable
{
public:
    SystemControlVariable();
    ~SystemControlVariable();
    double *params;
    double *reference;
    double *state;
    double *constraints;
    double *weightMatrix;
};*/

/*class QHP{
public:
    QHP();
    ~QHP();
    double *tensor_vector;
    double *column_vector;
};*/

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