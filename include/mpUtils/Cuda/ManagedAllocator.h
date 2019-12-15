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
#include <mpUtils/mpUtils.h>
#include "clionCudaHelper.h"
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
/**
 * class ManagedVector
 *
 * Behaves like an std vector but uses cuda unified / managed memory internally. Also adds some conversions and assignments.
 *
 * usage:
 * Use like std vector. Call getVectorReference() to use the vector data inside of a cuda kernel function.
 *
 */
template <typename T>
class ManagedVector : public std::vector<T,ManagedAllocator<T>>
{
public:
    // inherit all constructors
    using parent = std::vector<T,ManagedAllocator<T>>;
    using parent::parent;

    ManagedVector() : parent() {}

    // usage in device code over Vector reference
    VectorReference<T> getVectorReference() &; //!< allow creation of vectorReference only from lvalues
    VectorReference<const T> getVectorReference() const &; //!< create a vector reference to const from const lvalues

    // convert to / from std vector
    explicit ManagedVector(const std::vector<T>& vec); //!< upload data from a std vector
    explicit operator std::vector<T>() const; //!< download data into a newly constructed std vector
    void assign(const std::vector<T>& vec); //!< assign data from an std vector

    // assign data from device memory
    void assignFromDeviceMem(const T* first, int count); //!< copy count elements into the vector, first needs to point to global device memory

    // convert from device vector
    template<bool b>
    explicit ManagedVector(const DeviceVector<T,b>& vec); //!< construct from a device vector
    template<bool b>
    void assign(const DeviceVector<T,b>& vec); //!< assign values from a device vector
};

//-------------------------------------------------------------------
// some helper functions

template <typename T>
bool operator==(const ManagedAllocator<T>&, const ManagedAllocator<T>&) { return true; }

template <typename T>
bool operator!=(const ManagedAllocator<T>&, const ManagedAllocator<T>&) { return false; }

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

//-------------------------------------------------------------------
// function definitions of ManagedVector class

template <typename T>
VectorReference<T> ManagedVector<T>::getVectorReference()&
{
    return VectorReference<T>(parent::data(),parent::size());
}

template <typename T>
VectorReference<const T> ManagedVector<T>::getVectorReference() const&
{
    return VectorReference<const T>(parent::data(),parent::size());
}

template <typename T>
ManagedVector<T>::ManagedVector(const std::vector<T>& vec) : parent(vec.begin(),vec.end())
{
}

template <typename T>
ManagedVector<T>::operator std::vector<T>() const
{
    return std::vector<T>(parent::begin(),parent::end());
}

template <typename T>
void ManagedVector<T>::assign(const std::vector<T>& vec)
{
    parent::assign(vec.begin(),vec.end());
}

template <typename T>
void ManagedVector<T>::assignFromDeviceMem(const T* first, int count)
{
    parent::resize(count);
    assert_cuda(cudaMemcpy(parent::data(),first,count,cudaMemcpyDefault));
}

template <typename T>
template <bool b>
ManagedVector<T>::ManagedVector(const DeviceVector<T,b>& vec) : parent(vec.size())
{
    assert_cuda(cudaMemcpy(parent::data(),vec.data(),vec.size(),cudaMemcpyDefault));
}

template <typename T>
template <bool b>
void ManagedVector<T>::assign(const DeviceVector<T,b>& vec)
{
    parent::resize(vec.size());
    assert_cuda(cudaMemcpy(parent::data(),vec.data(),vec.size(),cudaMemcpyDefault));
}

}
#endif //MPUTILS_MANAGEDALLOCATOR_H
