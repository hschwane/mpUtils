/*
 * mpUtils
 * VectorReference.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the VectorReference class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_VECTORREFERENCE_H
#define MPUTILS_VECTORREFERENCE_H

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

//-------------------------------------------------------------------
/**
 * class template VectorReference
 *
 * Used in device code to reference memory stored in some vector-like container.
 * Note that the internal data of the vector is referenced.
 * This means when the vector is changed on the host side (eg capacity changes),
 * this change is not automatically visible (the reference might even become unsafe to access).
 * So be careful with concurrent access, especially when using functions that can change capacity.
 *
 * usage: When writing a kernel expect a VectorReference<T> instead of T* or VectorReference<const T> for read only. When calling the kernel use make_vectorReference()
 * to make a vector reference from a supported vector-like type.
 *
 */
template <typename T>
class VectorReference
{
public:
    // used types
    using value_type = T;
    using pointer = T*;

    CUDAHOSTDEV VectorReference() : m_data(nullptr), m_size(0) {}
    CUDAHOSTDEV VectorReference(T* data, int size) : m_data(data), m_size(size) {}

    CUDAHOSTDEV operator VectorReference<const T>() {return VectorReference<const T>(m_data,m_size); } //!< implicitly convert to a reference to const

    // access
    CUDAHOSTDEV T& operator[](int idx) {return m_data[idx];} //!< access an element in the buffer, "map" and "enableWrite" needs to be true
    CUDAHOSTDEV const T& operator[](int idx) const {return m_data[idx];} //!< access an element in the buffer readonly, "map" and needs to be true
    CUDAHOSTDEV T* data() { return m_data;} //!< direct data access
    CUDAHOSTDEV const T* data() const { return m_data;} //!< direct data access

    // iterators
    typedef T* iterator;
    typedef const T* const_iterator;

    CUDAHOSTDEV iterator begin() { return m_data;} //!< get an iterator to the beginning
    CUDAHOSTDEV iterator end() { return m_data+size();}//!< get an iterator to the end
    CUDAHOSTDEV const_iterator cbegin() const { return m_data;} //!< get a const iterator to the beginning
    CUDAHOSTDEV const_iterator cend() const { return m_data+size();} //!< get a const iterator to the end

    // check size
    CUDAHOSTDEV int size() const {return m_size;} //!< the number of elements of value_type stored in this vector
    CUDAHOSTDEV bool empty() const {return m_size >0;} //!< check if the original vector is empty

private:
    T* m_data;
    int m_size;
};

}

#endif //MPUTILS_VECTORREFERENCE_H
