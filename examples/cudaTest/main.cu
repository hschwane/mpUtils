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

inline __device__ int warpReduceSum(int val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK,val, offset);
    return val;
}

inline __device__ int blockReduceSum(int val)
{

    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__global__ void deviceReduceWarpAtomicKernel(int *in, int* out, int N)
{
    int sum = int(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < N;
        i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }

//    typedef cub::WarpReduce<int> WarpReduce;
//    __shared__ typename WarpReduce::TempStorage temp_storage[16];
//    int warp_id = threadIdx.x / 32;
//    sum = WarpReduce(temp_storage[warp_id]).Sum(sum);
//
    sum = warpReduceSum(sum);

    if ( (threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(out, sum);
}

__global__ void init(int *i, int* res, int N)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
        *res = 0;
    for( int idx : gridStrideRange(N))
    {
        i[idx] = 2;
    }
}

int main()
{
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("cudaTest", "0.9.1");

    int N = 32000;

    int *i;
    int *res;
    cudaMalloc(&i,N*sizeof(int));
    cudaMalloc(&res,sizeof(int));

    init<<<numBlocks(N,512),512>>>(i,res,N);

//    mpu::SimpleStopwatch sw;

    deviceReduceWarpAtomicKernel<<<numBlocks(N/4,512),512>>>(i,res,N);

//    cudaDeviceSynchronize()
//    sw.pause();

    int resCPU = 0;
    cudaMemcpy(&resCPU,res,sizeof(int),cudaMemcpyDeviceToHost);

    myLog.print(LogLvl::INFO) << "result: " << resCPU;


    return 0;
}