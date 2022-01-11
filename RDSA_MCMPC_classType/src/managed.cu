/* 
------2021.12.21 start making-------------
mallocとcudaMallocによるオペレーションでは、動的配列をメンバに持つデータ構造体を上手く定義できない
thrustも同様で、内部に動的な配列をメンバに持つような場合に上手く機能しない（恐らくコンストラクタ実行時にデバイス側とホスト側を
横断出来ていない？）

--->>> 上記は、ユニファイドメモリを使用することで解決できるかも知れない
その際、cudaMallocManaged関数を構造体とそのメンバ変数（動的な配列）に
それぞれ実行する必要がある
動的な配列は、それ自体をクラスとして処理すれば解決できる

managedクラスは、
newで宣言された構造体、あるいはクラスオブジェクトをユニファイドメモリに格納する関数で構成される
本来deleteを用いたオペレーションが必要になるが、
値渡しを用いて、構造体のメンバクラスのデストラクタを実行を繰り返し対応する（メモリリークに対する処置）
残ったやつはユニファイドメモリ上に存在するので、cudaDeviceReset関数でまとめて消せる
*/

#include "../include/DataStructures.cuh"

void* Managed::operator new(size_t len)
{
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    // printf("called Managed object!!\n");
    return ptr;
}

void* Managed::operator new[](size_t len)
{
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    // printf("called Managed object!!\n");
    return ptr;
}

void Managed::operator delete(void *ptr)
{
    cudaDeviceSynchronize();
    cudaFree(ptr);
}