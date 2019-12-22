/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utillities
 * A utility library for my personal c++ projects
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

/*
 * This is testing features of the matrix class... to be replaced by actual unit tests in the future...
 */

//#include <mpUtils/mpGraphics.h>
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpCuda.h>

#define FULL_MASK 0xffffffff

using namespace mpu;

struct ManagedData : mpu::Managed
{
    int i;
};

__global__ void init(mpu::VectorReference<int> data, ManagedData* res, int N)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
        res->i = 25;
    for( int idx : gridStrideRange(data.size()))
    {
        data[idx] += 2;
    }
}

int main()
{
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("cudaTest", "0.9.1");

    int N = 32000;


    mpu::DeviceVector<int> data(N,5);

    ManagedData *res = new ManagedData;

    init<<<numBlocks(N,512),512>>>(data.getVectorReference(), res,N);

//    mpu::SimpleStopwatch sw;
    assert_cuda(cudaDeviceSynchronize());
//    sw.pause();

//    int resCPU = 0;
//    cudaMemcpy(&resCPU,res,sizeof(int),cudaMemcpyDeviceToHost);

//    mpu::PinnedVector<int> hostData = data;

//    int i = 10;
//    const int& ref = i;

//    const  mpu::DeviceVector<int>& ref = data;


    DeviceVector<int> copy(30,10);
    copy[25] += 25;

    ManagedVector<int> managed(data);
    ManagedVector<int> managed2;
    managed2 = managed;


    myLog.print(LogLvl::INFO) << "result: " << res->i << " value[10]=" << data[10] << " copy[25]=" << copy[25] << " managed[13]= " << managed2[13];


    DeviceVector<float> a(128);
    DeviceVector<float> b(a);
    DeviceVector<float> c(16000);
    DeviceVector<float> d(32000);
    DeviceVector<float> e(64000);
    c=d;
    e=a;
    d=c;
    swap(a,e);

    return 0;
}