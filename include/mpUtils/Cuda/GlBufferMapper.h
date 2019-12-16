/*
 * mpUtils
 * GlBufferMapper.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the GlBufferMapper class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_GLBUFFERMAPPER_H
#define MPUTILS_GLBUFFERMAPPER_H

// includes
//--------------------
#include "clionCudaHelper.h"
#include "cudaUtils.h"
#include "mpUtils/Log/Log.h"
#include "VectorReference.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
/**
 * class GlBufferMapper
 *
 * Wrapper for cuda openGL interop. Maps an openGL buffer for use with cuda.
 *
 * usage:
 * Pass an openGL buffer ant set the apropriate type. Then you can map the buffer and use it in cuda code.
 * After usage call unmap to unmap the buffer again.
 *
 */
 template <typename T>
class GlBufferMapper
{
public:
    GlBufferMapper() = default;
    explicit GlBufferMapper(uint32_t glBufferId, cudaGraphicsMapFlags flags = cudaGraphicsMapFlagsNone);
    ~GlBufferMapper();

    // make non copyable but movable
    GlBufferMapper(const GlBufferMapper& other) = delete;
    GlBufferMapper& operator=(const GlBufferMapper& other) = delete;
    GlBufferMapper(GlBufferMapper&& other) noexcept : GlBufferMapper() {*this = std::move(other);}
    GlBufferMapper& operator=(GlBufferMapper&& other) noexcept;

    void map(); //!< map the resource for use from cuda api
    void unmap(); //!< unmap, so the resource can be used by opengl again

    T* data(); //!< get device pointer to the mapped data
    int size(); //!< size of the mapped buffer

    VectorReference<T> getVectorReference() &; //!< allow creation of vectorReference only from lvalues
    VectorReference<const T> getVectorReference() const &; //!< create a vector reference to const from const lvalues

private:
    T* m_mappedData{nullptr};
    int m_size{0};
    cudaGraphicsResource* m_graphicsResource{nullptr};
};

//-------------------------------------------------------------------

template <typename T>
GlBufferMapper<T>::GlBufferMapper(uint32_t glBufferId, cudaGraphicsMapFlags flags)
{
    assert_cuda(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, glBufferId, flags));
}

template <typename T>
GlBufferMapper<T>& GlBufferMapper<T>::operator=(GlBufferMapper&& other) noexcept
{
    using std::swap;
    swap(m_mappedData,other.m_mappedData);
    swap(m_size,other.m_size);
    swap(m_graphicsResource,other.m_graphicsResource);
    return *this;
}

template <typename T>
GlBufferMapper<T>::~GlBufferMapper()
{
    if(m_mappedData)
        unmap();
    assert_cuda(cudaGraphicsUnregisterResource(m_graphicsResource));
}

template <typename T>
void GlBufferMapper<T>::map()
{
    assert_true(m_graphicsResource,"GlBufferMapper","Register a Buffer bevore ")
    int bufferSize;
    assert_cuda(cudaGraphicsMapResources(1, &m_graphicsResource));
    assert_cuda(cudaGraphicsResourceGetMappedPointer((void **)&m_mappedData, &bufferSize, m_graphicsResource));
    m_size = bufferSize / sizeof(T);
}

template <typename T>
void GlBufferMapper<T>::unmap()
{
    if(m_mappedData)
        assert_cuda(cudaGraphicsUnmapResources(1, &m_graphicsResource));
    m_mappedData = nullptr;
}

template <typename T>
T* GlBufferMapper<T>::data()
{
    assert_true(m_mappedData, "GlBufferMapper", "Call map() before accessing the pointer or the size");
    return m_mappedData;
}

template <typename T>
int GlBufferMapper<T>::size()
{
    assert_true(m_mappedData, "GlBufferMapper", "Call map() before accessing the pointer or the size");
    return m_size;
}

template <typename T>
VectorReference<T> GlBufferMapper<T>::getVectorReference()&
{
    return VectorReference<T>(data(),size());
}

template <typename T>
VectorReference<const T> GlBufferMapper<T>::getVectorReference() const&
{
    return VectorReference<const T>(data(),size());
}

}
#endif //MPUTILS_GLBUFFERMAPPER_H
