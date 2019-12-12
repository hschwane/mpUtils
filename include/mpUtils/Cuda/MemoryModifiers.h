/*
 * mpUtils
 * Managed.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MEMORYMODIFIERS_H
#define MPUTILS_MEMORYMODIFIERS_H

// includes
//--------------------
#include "clionCudaHelper.h"
#include "mpUtils/external/cuda/helper_math.h"
#include "cudaUtils.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

/**
 * @brief Deriving from this class will ensure every instance of the derived class is created using cuda managed memory
 */
class Managed
{
public:
    void* operator new(size_t len)
    {
        void* ptr;
        assert_cuda(cudaMallocManaged(&ptr, len));
        return ptr;
    }

    void* operator new[](size_t len)
    {
        void* ptr;
        assert_cuda(cudaMallocManaged(&ptr, len));
        return ptr;
    }

    void operator delete(void* ptr)
    {
        assert_cuda(cudaFree(ptr));
    }
    void operator delete[](void* ptr)
    {
        assert_cuda(cudaFree(ptr));
    }
};

/**
 * @brief Deriving from this class will ensure every instance of the derived class is created using cuda pinned memory
 */
class Pinned
{
public:
    void* operator new(size_t len)
    {
        void* ptr;
        assert_cuda(cudaMallocHost(&ptr, len));
        return ptr;
    }

    void* operator new[](size_t len)
    {
        void* ptr;
        assert_cuda(cudaMallocHost(&ptr, len));
        return ptr;
    }

    void operator delete(void* ptr)
    {
        assert_cuda(cudaFree(ptr));
    }

    void operator delete[](void* ptr)
    {
        assert_cuda(cudaFree(ptr));
    }
};

}

#endif //MPUTILS_MEMORYMODIFIERS_H
