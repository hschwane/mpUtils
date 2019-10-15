/*
 * mpUtils
 * Buffer.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Opengl/Buffer.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

BufferBase::BufferBase()
{
    glCreateBuffers(1,&m_bufferHandle);
}

BufferBase::~BufferBase()
{
    glDeleteBuffers(1,&m_bufferHandle);
}

BufferBase::operator uint32_t() const
{
    return m_bufferHandle;
}

BufferBase& BufferBase::operator=(BufferBase&& other) noexcept
{
    using std::swap;
    swap(m_bufferHandle,other.m_bufferHandle);
    return *this;
}

void BufferBase::bindBase(uint32_t binding, GLenum target) const
{
    glBindBufferBase(target, binding, *this);
}

}}