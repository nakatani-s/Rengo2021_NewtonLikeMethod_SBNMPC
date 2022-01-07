/* 
------2021.12.21 start making-------------
*/

#include <curand_kernel.h>
#include "myController.cuh"

#ifndef DATASTRUCTURE_CUH
#define DATASTRUCTURE_CUH
class SampleInfo
{
public:
    SampleInfo(); //コンストラクタ
    ~SampleInfo(); //デストラクタ
    double cost;
    double weight;
    double *inputSeq;
};
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

class QHP{
public:
    QHP();
    ~QHP();
    double *tensor_vector;
    double *column_vector;
};

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