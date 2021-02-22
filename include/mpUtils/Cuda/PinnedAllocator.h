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
#include "cudaUtils.h"
#include "VectorReference.h"
#include "DeviceVector.h"
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
/**
 * class PinnedVector
 *
 * Behaves like an std vector but uses cuda unified / managed memory internally. Also adds some conversions and assignments.
 *
 * usage:
 * Use like std vector. Call getVectorReference() to use the vector data inside of a cuda kernel function.
 *
 */
template <typename T>
class PinnedVector : public std::vector<T,PinnedAllocator<T>>
{
public:
    // inherit all constructors
    using parent = std::vector<T, PinnedAllocator < T>>;
    using parent::parent;

    PinnedVector() : parent() {}

    // convert to / from std vector
    explicit PinnedVector(const std::vector<T>& vec); //!< upload data from a std vector
    explicit operator std::vector<T>() const; //!< download data into a newly constructed std vector
    void assign(const std::vector<T>& vec); //!< assign data from an std vector

    // assign data from device memory
    void assignFromDeviceMem(const T* first, int count); //!< copy count elements into the vector, first needs to point to global device memory

    // convert from device vector
    template <bool b>
    explicit PinnedVector(const DeviceVector <T, b>& vec); //!< construct from a device vector
    template <bool b>
    void assign(const DeviceVector <T, b>& vec); //!< assign values from a device vector
};

//-------------------------------------------------------------------
// some helper functions and aliases

template <typename T>
bool operator==(const PinnedAllocator<T>&, const PinnedAllocator<T>&) { return true; }

template <typename T>
bool operator!=(const PinnedAllocator<T>&, const PinnedAllocator<T>&) { return false; }

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
    assert_cuda(cudaFreeHost(p));
}

//-------------------------------------------------------------------
// function definitions of pinnedVector class

template <typename T>
PinnedVector<T>::PinnedVector(const std::vector<T>& vec) : parent(vec.begin(),vec.end())
{
}

template <typename T>
PinnedVector<T>::operator std::vector<T>() const
{
    return std::vector<T>(parent::begin(),parent::end());
}

template <typename T>
void PinnedVector<T>::assign(const std::vector<T>& vec)
{
    parent::assign(vec.begin(),vec.end());
}

template <typename T>
void PinnedVector<T>::assignFromDeviceMem(const T* first, int count)
{
    parent::resize(count);
    assert_cuda(cudaMemcpy(parent::data(),first,count,cudaMemcpyDeviceToHost));
}

template <typename T>
template <bool b>
PinnedVector<T>::PinnedVector(const DeviceVector<T,b>& vec) : parent(vec.size())
{
    assert_cuda(cudaMemcpy(parent::data(),vec.first(),vec.size(),cudaMemcpyDeviceToHost));
}

template <typename T>
template <bool b>
void PinnedVector<T>::assign(const DeviceVector<T,b>& vec)
{
    parent::resize(vec.size());
    assert_cuda(cudaMemcpy(parent::data(),vec.first(),vec.size(),cudaMemcpyDeviceToHost));
}

}
#endif //MPUTILS_PINNEDALLOCATOR_H
