/*
 * mpUtils
 * ManagedAllocator.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the ManagedAllocator class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_MANAGEDALLOCATOR_H
#define MPUTILS_MANAGEDALLOCATOR_H

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
 * class ManagedAllocator
 *
 * std::allocator compatible class to allocate and deallocate Managed Memory.
 *
 * usage:
 * Use with std container or use the typedef ManagedVector below.
 *
 */
template <typename T>
class ManagedAllocator
{
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    ManagedAllocator() = default;

    template <typename U>
    ManagedAllocator(const ManagedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n);
    void deallocate(T* p, std::size_t);
};

//-------------------------------------------------------------------
// some helper functions and aliases

template <typename T>
bool operator==(const ManagedAllocator<T>&, const ManagedAllocator<T>&) { return true; }

template <typename T>
bool operator!=(const ManagedAllocator<T>&, const ManagedAllocator<T>&) { return false; }

/**
 * @brief specialization of std vector, but memory is allocated as cuda managed memory
 */
template <typename T>
using ManagedVector = std::vector<T,ManagedAllocator<T>>;

/**
 * @brief Creates a vector reference for use on the device from a managed vector
 * @param vec the managed vector
 * @returns the VectorReference which references the device vector
 */
template <typename T>
VectorReference<T> make_vectorReference(ManagedVector<T>& vec)
{
    return VectorReference<T>(vec.data(),vec.size());
}

/**
 * @brief Creates a vector reference to const for use on the device from a const managed vector
 * @param vec the managed vector
 * @returns the VectorReference which references the device vector
 */
template <typename T>
VectorReference<const T> make_vectorReference(const ManagedVector<T>& vec)
{
    return VectorReference<const T>(vec.data(),vec.size());
}

//-------------------------------------------------------------------
// function definitions of managed allocator class

template <typename T>
T* ManagedAllocator<T>::allocate(std::size_t n)
{
    T* ptr;
    assert_cuda(cudaMallocManaged(&ptr, sizeof(T)*n));
    return ptr;
}

template <typename T>
void ManagedAllocator<T>::deallocate(T* p, std::size_t)
{
    assert_cuda(cudaFree(p));
}

}
#endif //MPUTILS_MANAGEDALLOCATOR_H
