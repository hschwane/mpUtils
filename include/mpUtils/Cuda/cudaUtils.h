/*
 * mpUtils
 * cudaUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_CUDAUTILS_H
#define MPUTILS_CUDAUTILS_H

// includes
//--------------------
#include "clionCudaHelper.h"
#include "mpUtils/external/cuda/helper_math.h"
#include "mpUtils/Log/Log.h"
#include "mpUtils/Misc/Range.h"
#include "mpUtils/external/gcem/gcem.hpp"
//--------------------

// some defines
//--------------------
//!< wrap cuda function calls in this to check for errors
#define assert_cuda(CODE) mpu::_cudaAssert((CODE),MPU_FILEPOS)
//!< use this to define a function as usable on host and device
#ifndef CUDAHOSTDEV
#define CUDAHOSTDEV __host__ __device__
#endif
//--------------------

namespace mpu {

/**
 * @brief  called by the macro above to evaluate cuda errors
 */
inline void _cudaAssert(cudaError_t code, std::string &&filepos)
{
    if(code != cudaSuccess)
    {
        std::string message("Cuda error: " + std::string(cudaGetErrorString(code)));

        if(!(mpu::Log::noGlobal()))
        {
            if(mpu::Log::getGlobal().getLogLevel() >= mpu::LogLvl::FATAL_ERROR)
                mpu::Log::getGlobal()(mpu::LogLvl::FATAL_ERROR, std::move(filepos), "cuda") << message;
            mpu::Log::getGlobal().flush();
        }

        throw std::runtime_error(message);
    }
}

/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the grid size from the data size
 *          indices will run in the range [0,problemSize)
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> gridStrideRange(int problemSize)
{
    return Range<int>(blockIdx.x * blockDim.x + threadIdx.x, problemSize, gridDim.x * blockDim.x);
}

/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the grid size from the data size
 *          indices will run in the range [firstElement,problemSize)
 * @param firstElement the first element of the dataset to process
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> gridStrideRange(int firstElement, int problemSize)
{
    firstElement += blockIdx.x * blockDim.x + threadIdx.x;
    return Range<int>(firstElement, problemSize, gridDim.x * blockDim.x);
}

/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the grid size from the data size
 *          in y direction. indices will run in the range [0,problemSize)
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> gridStrideRangeY(int problemSize)
{
    return Range<int>(blockIdx.y * blockDim.y + threadIdx.y, problemSize, gridDim.y * blockDim.y);
}

/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the grid size from the data size
 *          in y direction. indices will run in the range [firstElement,problemSize)
 * @param firstElement the first element of the dataset to process
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> gridStrideRangeY(int firstElement, int problemSize)
{
    firstElement += blockIdx.y * blockDim.y + threadIdx.y;
    return Range<int>(firstElement, problemSize, gridDim.y * blockDim.y);
}

/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the grid size from the data size
 *          in z direction. indices will run in the range [0,problemSize)
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> gridStrideRangeZ(int problemSize)
{
    return Range<int>(blockIdx.z * blockDim.z + threadIdx.z, problemSize, gridDim.z * blockDim.z);
}

/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the grid size from the data size
 *          in z direction. indices will run in the range [firstElement,problemSize)
 * @param firstElement the first element of the dataset to process
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> gridStrideRangeZ(int firstElement, int problemSize)
{
    firstElement += blockIdx.z * blockDim.z + threadIdx.z;
    return Range<int>(firstElement, problemSize, gridDim.z * blockDim.z);
}

/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the block size from the data size
 *          indices will run in the range [0,problemSize)
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> blockStrideRange(int problemSize)
{
    return Range<int>(threadIdx.x, problemSize, blockDim.x);
}

/**
 * @brief generates a Range to be used in a for each loop inside a kernel to decouple the block size from the data size
 *          indices will run in the range [firstElement,problemSize)
 * @param firstElement the first element of the dataset to process
 * @param problemSize the size of the data to process
 * @return a mpu::Range object to be used inside a for each loop
 */
inline __device__ Range<int> blockStrideRange(int firstElement, int problemSize)
{
    firstElement += threadIdx.x;
    return Range<int>(firstElement, problemSize, blockDim.x);
}

/**
 * @brief calculates the number of cuda blocks to be launched
 * @param problemSize number of things that needs to be published
 * @return the number of cuda blocks that should be launched
 */
constexpr size_t numBlocks(size_t problemSize, size_t blockSize)
{
    return gcem::max((problemSize + blockSize - 1) / blockSize, 1);
}

/**
 * @brief Uploads data to gpu memory.
 * @param dest pointer to gpu memory where data is written
 * @param src pointer to cpu memory where data is read
 * @param count number of elements (NOT bytes) of type T to be copied
 */
template <typename T>
void cudaUpload(T* dst, const T* src, size_t count)
{
    assert_cuda(cudaMemcpy(dst,src,count* sizeof(T),cudaMemcpyHostToDevice));
}

/**
 * @brief Uploads data to gpu memory.
 * @param dest pointer to gpu memory where data is written
 * @param src pointer to cpu memory where data is read
 * @param count number of elements (NOT bytes) of type T to be copied
 */
template <typename T>
void cudaDownload(T* dst, const T* src, size_t count)
{
    assert_cuda(cudaMemcpy(dst,src,count* sizeof(T),cudaMemcpyDeviceToHost));
}

/**
 * @brief Uploads data to gpu memory.
 * @param dest pointer to gpu memory where data is written
 * @param src pointer to cpu memory where data is read
 * @param count number of elements (NOT bytes) of type T to be copied
 */
template <typename T>
void cudaCopy(T* dst, const T* src, size_t count)
{
    assert_cuda(cudaMemcpy(dst,src,count* sizeof(T),cudaMemcpyDeviceToDevice));
}

}

//--------------------
// stream operators for cuda (outside of mpu namespace)

inline std::ostream & operator<<(std::ostream & stream, const float2 & item)
{
    stream << "(" << item.x << "," << item.y << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const float3 & item)
{
    stream << "(" << item.x << "," << item.y << "," << item.z << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const float4 & item)
{
    stream << "(" << item.x << "," << item.y << "," << item.z << "," << item.w << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const double2 & item)
{
    stream << "(" << item.x << "," << item.y << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const double3 & item)
{
    stream << "(" << item.x << "," << item.y << "," << item.z << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const double4 & item)
{
    stream << "(" << item.x << "," << item.y << "," << item.z << "," << item.w << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const int2 & item)
{
    stream << "(" << item.x << "," << item.y << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const int3 & item)
{
    stream << "(" << item.x << "," << item.y << "," << item.z << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const int4 & item)
{
    stream << "(" << item.x << "," << item.y << "," << item.z << "," << item.w << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const uint2 & item)
{
    stream << "(" << item.x << "," << item.y << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const uint3 & item)
{
    stream << "(" << item.x << "," << item.y << "," << item.z << ")";
    return stream;
}

inline std::ostream & operator<<(std::ostream & stream, const uint4 & item)
{
    stream << "(" << item.x << "," << item.y << "," << item.z << "," << item.w << ")";
    return stream;
}

#endif //MPUTILS_CUDAUTILS_H
