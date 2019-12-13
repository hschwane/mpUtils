/*
 * mpUtils
 * PinnedAllocator.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the PinnedAllocator class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_PINNEDALLOCATOR_H
#define MPUTILS_PINNEDALLOCATOR_H

// includes
//--------------------
#include <type_traits>
#include "clionCudaHelper.h"
#include "cudaUtils.h"
#include "VectorReference.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
/**
 * class PinnedAllocator
 *
 * std::allocator compatible class to allocate and deallocate Pinned Memory.
 *
 * usage:
 * Use with std container or use the typedef PinnedVector below.
 *
 */
template <typename T>
class PinnedAllocator
{
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    PinnedAllocator() = default;

    template <typename U>
    PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n);
    void deallocate(T* p, std::size_t);
};

//-------------------------------------------------------------------
// some helper functions and aliases

template <typename T>
bool operator==(const PinnedAllocator<T>&, const PinnedAllocator<T>&) { return true; }

template <typename T>
bool operator!=(const PinnedAllocator<T>&, const PinnedAllocator<T>&) { return false; }

/**
 * @brief specialization of std vector, but memory is allocated as cuda pinned memory
 */
template <typename T>
using PinnedVector = std::vector<T,PinnedAllocator<T>>;

//!< create a pinned vector from a std vector
template <typename T>
PinnedVector<T> make_pinnedVector(const std::vector<T>& v)
{
    return PinnedVector<T>(v.begin(),v.end());
}

//!< create a std vector from a pinned vector
template <typename T>
std::vector<T> to_stdvector(const PinnedVector<T>& v)
{
    return std::vector<T>(v.begin(),v.end());
}

//-------------------------------------------------------------------
// function definitions of managed allocator class

template <typename T>
T* PinnedAllocator<T>::allocate(std::size_t n)
{
    T* ptr;
    assert_cuda(cudaMallocHost(&ptr, sizeof(T)*n));
    return ptr;
}

template <typename T>
void PinnedAllocator<T>::deallocate(T* p, std::size_t)
{
    assert_cuda(cudaFree(p));
}


}
#endif //MPUTILS_PINNEDALLOCATOR_H
