/*
 * mpUtils
 * Buffer.h
 *
 * Contains the VertexArrray class to manage an openGL VertexArrayObject
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2017 Hendrik Schwanekamp
 *
 * This file was originally written and generously provided for this framework from Johannes Braun.
 *
 */
#pragma once

#include <cinttypes>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include "Buffer.h"

namespace mpu {
namespace gph {

class VertexArray
{
public:
	VertexArray(); //!< default constructor

    ~VertexArray(); //!< destructor
    operator uint32_t() const; //!< convert to opengl handle for use in raw opengl functions

    // make vao non copyable but movable
    VertexArray(const VertexArray& other) = delete;
    VertexArray& operator=(const VertexArray& other) = delete;
    VertexArray(VertexArray&& other) noexcept : m_vaoHandle(0){*this = std::move(other);};
    VertexArray& operator=(VertexArray&& other) noexcept {using std::swap; swap(m_vaoHandle,other.m_vaoHandle); return *this;};


    void enableArray(GLuint attribIndex); //!< enable the attibute array "attribIndex"
    void disableArray(GLuint attribIndex); //!< disable the attibute array "attribIndex"

    /**
     * @brief Assign an openGL Buffer to an index on the vao
     * @param index the index where the buffer is added
     * @param buffer the buffer to add to the vao
     * @param offset the offset in bytes from the beginning of the buffer
     * @param stride byte offset from one vertex to the next (size of the vertex in bytes)
     */
    void addBuffer(GLuint bindingIndex, const BufferBase& buffer, GLintptr offset, GLsizei stride);

    /**
     * @brief Set the format of an attribute that uses float or vec inside the shader.
     *          When passing integer or double data these will be converted to float. Use configureAttribFormatInt()
     *          or configureAttribFormatDouble() to prevent conversion.
     * @param attribIndex index of the attribute
     * @param vecSize size of the vector per vertex (1-4)
     * @param relativeOffset the relative offset of the vector inside the buffer. Use offset_of(&MyVertex::my_member)
     * @param type the type of the input data in the buffer, GL_FLOAT mostly. Integer and doubles are converted into floats.
     * @param normalize enable normalization when passing integers or unsigned integers that are converted to float
     */
    void configureAttribFormat(GLuint attribIndex, GLint vecSize, GLuint relativeOffset, GLenum type=GL_FLOAT, GLboolean normalize = GL_FALSE);

    /**
     * @brief Set the format of an attribute that uses int, uint, ivec or uvec inside the shader.
     * @param attribIndex index of the attribute
     * @param vecSize size of the vector per vertex (1-4)
     * @param relativeOffset the relative offset of the vector inside the buffer. Use offset_of(&MyVertex::my_member)
     * @param type the type of the input data in the buffer, GL_INT,  GL_UNSIGNED_INT, usw
     */
    void configureAttribFormatInt(GLuint attribIndex, GLint vecSize, GLuint relativeOffset, GLenum type=GL_INT);

    /**
     * @brief Set the format of an attribute that uses double or dvec inside the shader. Data in the buffer should be doubles.
     * @param attribIndex index of the attribute
     * @param vecSize size of the vector per vertex (1-4)
     * @param relativeOffset the relative offset of the vector inside the buffer. Use offset_of(&MyVertex::my_member)
     */
    void configureAttribFormatDouble(GLuint attribIndex, GLint vecSize, GLuint relativeOffset);

    void addBinding( GLuint attribIndex, GLuint bindingIndex); //!< configure a buffer to be the data source for an attribute array

    /**
     * @brief Enable a attribute index, configure it and set the buffer at bindingIndex as the data source.
     *          Use for attributes that use float or vec in the shader. When passing integer or double data,
     *          these will be converted to float. Use configureAttribFormatInt()
     *          or configureAttribFormatDouble() to prevent conversion.
     * @param attribIndex index of the attribute
     * @param bindingIndex binding index of the data source buffer
     * @param vecSize size of the vector per vertex (1-4)
     * @param relativeOffset the relative offset of the vector inside the buffer. Use offset_of(&MyVertex::my_member)
     * @param type the type of the input data in the buffer, GL_FLOAT mostly. Integer and doubles are converted into floats.
     * @param normalize enable normalization when passing integers or unsigned integers that are converted to float
     */
    void addAttributeArray(GLuint attribIndex, GLuint bindingIndex, GLint vecSize, GLuint relativeOffset, GLenum type=GL_FLOAT, GLboolean normalize = GL_FALSE);

    /**
     * @brief Enable a attribute index, configure it and set the buffer at bindingIndex as the data source.
     *          Use for attributes that use int, uint, ivec or uvec inside the shader.
     * @param attribIndex index of the attribute
     * @param bindingIndex binding index of the data source buffer
     * @param vecSize size of the vector per vertex (1-4)
     * @param relativeOffset the relative offset of the vector inside the buffer. Use offset_of(&MyVertex::my_member)
     * @param type the type of the input data in the buffer, GL_INT,  GL_UNSIGNED_INT, usw
     */
    void addAttributeArrayInt(GLuint attribIndex, GLuint bindingIndex, GLint vecSize, GLuint relativeOffset, GLenum type=GL_INT);

    /**
     * @brief Enable a attribute index, configure it and set the buffer at bindingIndex as the data source.
     *          Use for attributes that use double or dvec inside the shader. Data in the buffer should be doubles.
     * @param attribIndex index of the attribute
     * @param bindingIndex binding index of the data source buffer
     * @param vecSize size of the vector per vertex (1-4)
     * @param relativeOffset the relative offset of the vector inside the buffer. Use offset_of(&MyVertex::my_member)
     */
    void addAttributeArrayDouble(GLuint attribIndex, GLuint bindingIndex, GLint vecSize, GLuint relativeOffset);

    /**
     * @brief Enable a atrribute index and configure it. Bind "buffer" at bindingIndex and set it as the data source.
     *          Use for attributes that use float or vec in the shader. When passing integer or double data,
     *          these will be converted to float. Use configureAttribFormatInt()
     *          or configureAttribFormatDouble() to prevent conversion.
     * @param attribIndex index of the attribute
     * @param bindingIndex binding index of the data source buffer
     * @param buffer the buffer object to bind on the binding index
     * @param offset the offset in bytes from the beginning of the buffer
     * @param stride byte offset from one vertex to the next (size of the vertex in bytes)
     * @param vecSize size of the vector per vertex (1-4)
     * @param relativeOffset the relative offset of the vector inside the buffer. Use offset_of(&MyVertex::my_member)
     * @param type the type of the input data, GL_FLOAT mostly. Integer are converted into floating point
     * @param normalize enable normalization when passing integers or unsigned integers that are converted to float
     */
    void addAttributeBufferArray(GLuint attribIndex, GLuint bindingIndex, const BufferBase& buffer, GLintptr offset, GLsizei stride, GLint vecSize, GLuint relativeOffset=0, GLenum type=GL_FLOAT, GLboolean normalize = GL_FALSE);

    /**
     * @brief Enable a atrribute index and configure it. Bind "buffer" at bindingIndex and set it as the data source.
     *          Use for attributes that use int, uint, ivec or uvec inside the shader.
     * @param attribIndex index of the attribute
     * @param bindingIndex binding index of the data source buffer
     * @param buffer the buffer object to bind on the binding index
     * @param offset the offset in bytes from the beginning of the buffer
     * @param stride byte offset from one vertex to the next (size of the vertex in bytes)
     * @param vecSize size of the vector per vertex (1-4)
     * @param relativeOffset the relative offset of the vector inside the buffer. Use offset_of(&MyVertex::my_member)
     * @param type the type of the input data in the buffer, GL_INT,  GL_UNSIGNED_INT, usw
     */
    void addAttributeBufferArrayInt(GLuint attribIndex, GLuint bindingIndex, const BufferBase& buffer, GLintptr offset, GLsizei stride, GLint vecSize, GLuint relativeOffset, GLenum type=GL_INT);

    /**
     * @brief Enable a atrribute index and configure it. Bind "buffer" at bindingIndex and set it as the data source.
     *          Use for attributes that use double or dvec inside the shader. Data in the buffer should be doubles.
     * @param attribIndex index of the attribute
     * @param bindingIndex binding index of the data source buffer
     * @param buffer the buffer object to bind on the binding index
     * @param offset the offset in bytes from the beginning of the buffer
     * @param stride byte offset from one vertex to the next (size of the vertex in bytes)
     * @param vecSize size of the vector per vertex (1-4)
     * @param relativeOffset the relative offset of the vector inside the buffer. Use offset_of(&MyVertex::my_member)
     */
    void addAttributeBufferArrayDouble(GLuint attribIndex, GLuint bindingIndex, const BufferBase& buffer, GLintptr offset, GLsizei stride, GLint vecSize, GLuint relativeOffset);

    void setIndexBuffer( const BufferBase& buffer) const; //!< set a buffer as index buffer for the vao

	void bind() const; //!< bind the vao to use it as the rendering source

private:
    uint32_t m_vaoHandle;
};


inline VertexArray::VertexArray()
{
    glCreateVertexArrays(1,&m_vaoHandle);
}

inline VertexArray::~VertexArray()
{
    glDeleteVertexArrays(1,&m_vaoHandle);
}

inline VertexArray::operator uint32_t() const
{
    return m_vaoHandle;
}

inline void VertexArray::enableArray(const GLuint attribIndex)
{
    glEnableVertexArrayAttrib(*this, attribIndex);
}

inline void VertexArray::disableArray(GLuint attribIndex)
{
    glDisableVertexArrayAttrib(*this, attribIndex);
}

inline void VertexArray::addBuffer(const GLuint bindingIndex, const BufferBase& buffer, const GLintptr offset, const GLsizei stride)
{
    glVertexArrayVertexBuffer(*this, bindingIndex, buffer, offset, stride);
}

inline void VertexArray::configureAttribFormat(const GLuint attribIndex, const GLint vecSize, const GLuint relativeOffset, const GLenum type, const GLboolean normalize)
{
    glVertexArrayAttribFormat(*this, attribIndex, vecSize, type, normalize, relativeOffset);
}

inline void VertexArray::configureAttribFormatInt(GLuint attribIndex, GLint vecSize, GLuint relativeOffset, GLenum type)
{
    glVertexArrayAttribIFormat(*this, attribIndex, vecSize, type, relativeOffset);
}

inline void VertexArray::configureAttribFormatDouble(GLuint attribIndex, GLint vecSize, GLuint relativeOffset)
{
    glVertexArrayAttribLFormat(*this, attribIndex, vecSize, GL_DOUBLE, relativeOffset);
}

inline void VertexArray::addBinding(GLuint attribIndex, GLuint bindingIndex)
{
    glVertexArrayAttribBinding(*this, attribIndex, bindingIndex);
}

inline void VertexArray::addAttributeArray(GLuint attribIndex, GLuint bindingIndex, GLint vecSize, GLuint relativeOffset,
                                    GLenum type, GLboolean normalize)
{
    enableArray(attribIndex);
    configureAttribFormat(attribIndex, vecSize, relativeOffset, type, normalize);
    addBinding(attribIndex, bindingIndex);
}

inline void VertexArray::addAttributeArrayInt(GLuint attribIndex, GLuint bindingIndex, GLint vecSize, GLuint relativeOffset,
                                           GLenum type)
{
    enableArray(attribIndex);
    configureAttribFormatInt(attribIndex, vecSize, relativeOffset, type);
    addBinding(attribIndex, bindingIndex);
}

inline void VertexArray::addAttributeArrayDouble(GLuint attribIndex, GLuint bindingIndex, GLint vecSize, GLuint relativeOffset)
{
    enableArray(attribIndex);
    configureAttribFormatDouble(attribIndex, vecSize, relativeOffset);
    addBinding(attribIndex, bindingIndex);
}

inline void VertexArray::addAttributeBufferArray(GLuint attribIndex, GLuint bindingIndex, const BufferBase& buffer, GLintptr offset,
                                                 GLsizei stride, GLint vecSize, GLuint relativeOffset, GLenum type,
                                                 GLboolean normalize)
{
    enableArray(attribIndex);
    addBuffer(bindingIndex, buffer, offset, stride);
    configureAttribFormat(attribIndex, vecSize, relativeOffset, type, normalize);
    addBinding(attribIndex,bindingIndex);
}

inline void VertexArray::addAttributeBufferArrayInt(GLuint attribIndex, GLuint bindingIndex, const BufferBase& buffer, GLintptr offset,
                                                    GLsizei stride, GLint vecSize, GLuint relativeOffset, GLenum type)
{
    enableArray(attribIndex);
    addBuffer(bindingIndex, buffer, offset, stride);
    configureAttribFormatInt(attribIndex, vecSize, relativeOffset, type);
    addBinding(attribIndex,bindingIndex);
}

inline void VertexArray::addAttributeBufferArrayDouble(GLuint attribIndex, GLuint bindingIndex, const BufferBase& buffer, GLintptr offset,
                                                       GLsizei stride, GLint vecSize, GLuint relativeOffset)
{
    enableArray(attribIndex);
    addBuffer(bindingIndex, buffer, offset, stride);
    configureAttribFormatDouble(attribIndex, vecSize, relativeOffset);
    addBinding(attribIndex,bindingIndex);
}

inline void VertexArray::setIndexBuffer(const BufferBase& buffer) const
{
	glVertexArrayElementBuffer(*this, buffer);
}

inline void VertexArray::bind() const
{
	glBindVertexArray(*this);
}


}}