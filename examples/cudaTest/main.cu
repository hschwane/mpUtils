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

__global__ void init(ManagedData *v, ManagedData* res, int N)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
        res->i = 25;
    for( int idx : gridStrideRange(N))
    {
        v[idx].i = 2;
    }
}

int main()
{
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("cudaTest", "0.9.1");

    int N = 32000;

    ManagedData *i = new ManagedData[N];
    ManagedData *res = new ManagedData;
//    cudaMalloc(&i,N*sizeof(int));
//    cudaMalloc(&res,sizeof(int));

    init<<<numBlocks(N,512),512>>>(i,res,N);

//    mpu::SimpleStopwatch sw;
    assert_cuda(cudaDeviceSynchronize());
//    sw.pause();

    int resCPU = 0;
//    cudaMemcpy(&resCPU,res,sizeof(int),cudaMemcpyDeviceToHost);

    myLog.print(LogLvl::INFO) << "result: " << res->i;


    return 0;
}