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
        data[idx] = 2;
    }
}

int main()
{
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("cudaTest", "0.9.1");

    int N = 32000;

    mpu::ManagedVector<int> data(N);

    ManagedData *res = new ManagedData;

    init<<<numBlocks(N,512),512>>>( make_vectorReference(data), res,N);

//    mpu::SimpleStopwatch sw;
    assert_cuda(cudaDeviceSynchronize());
//    sw.pause();

    int resCPU = 0;
//    cudaMemcpy(&resCPU,res,sizeof(int),cudaMemcpyDeviceToHost);

//    mpu::PinnedVector<int> hostData = data;

    myLog.print(LogLvl::INFO) << "result: " << res->i << " value[10] " << data[10];




    return 0;
}