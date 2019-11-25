/*
 * mpUtils
 * Buffer.h
 *
 * Contains the Buffer class to manage an openGL Buffer
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#pragma once

// includes
//--------------------
#include <GL/glew.h>
#include <utility>
#include <vector>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

/**
 * class BufferBase
 *
 * BufferBase class manages the livetime of an opengl buffer. Do not use directly.
 * Use "Buffer" instead.
 *
 */
class BufferBase
{
public:
    BufferBase();

    virtual ~BufferBase(); //!< destructor
    operator uint32_t() const; //!< convert to opengl handle for use in raw opengl functions

    // make Buffer non copyable but movable
    BufferBase(const BufferBase& other) = delete;
    BufferBase& operator=(const BufferBase& other) = delete;
    BufferBase(BufferBase&& other) noexcept : m_bufferHandle(0){ *this = std::move(other);};
    BufferBase& operator=(BufferBase&& other) noexcept;

    /**
     * Binds the buffer to the given fixed binding index and Target.
     * Possible targets: GL_ATOMIC_COUNTER_BUFFER, GL_TRANSFORM_FEEDBACK_BUFFER,
     *                      GL_UNIFORM_BUFFER or GL_SHADER_STORAGE_BUFFER
     * Binding is a number.
     * Refers to GLSL via:
     * layout(binding = N) <buffertype>...
     */
    void bindBase( uint32_t binding, GLenum target) const;

private:
    uint32_t m_bufferHandle;
};

//-------------------------------------------------------------------
/**
 * class Buffer
 *
 * usage:
 * Represents an openGL Buffer object. Template parameter T controls the type of data that is stored.
 * If you want to have access to the write() functions you need to set enable write to true.
 * Iterators as well as element access via operator[] are only available when setting map to true.
 * This will make use of a persistent coherent buffer mapping internally. Don't forget to sync to wait for previous
 * openGL calls to complete that use on the buffer.
 *
 * Use bindBase() to bind the buffer for usage in openGL.
 * There are also some utility functions available to copy the buffer or create a clone or invalidate the Buffer.
 *
 */
template <typename T, bool enableWrite = false, bool map = false>
class Buffer : public BufferBase
{
public:
    explicit Buffer(size_t size);
    explicit Buffer(std::vector<T> data);
    Buffer(T* data, size_t count);

    Buffer& operator=(Buffer&& other) = default;
    Buffer(Buffer&& other) = default;

    ~Buffer() override;

    /**
     * @brief Copy data from another buffer into this buffer
     * @param source the buffer where data is copied from
     * @param count number of elements to be copied. pass 0 to copy entire buffer
     * @param sourceOffset offset in the source buffer
     * @param destOffset offset in the destination (this) buffer
     */
    template <bool isWrite, bool isMapped>
    void copy(const Buffer<T,isWrite,isMapped>& source, size_t count =0, size_t sourceOffset=0, size_t destOffset=0);

    /**
     * @brief create a clone of this buffer
     * @return the new buffer
     */
    Buffer clone();

    /**
     * @brief write data from an std vector to the buffer
     * @param data the vector used as the source for the data
     * @param offset offset inside the buffer where writing starts
     */
    void write( std::vector<T> data, size_t offset = 0);

    /**
     * @brief write data to the buffer
     * @param data pointer to the data source
     * @param count number of elements to write into the buffer
     * @param offset offset inside the buffer where writing starts
     */
    void write( T* data, size_t count, size_t offset = 0);

    /**
     * @brief read data from the buffer into a std vector
     * @param count number of elements to read
     * @param offset offset where to start reading
     * @return a vector containing the
     */
    std::vector<T> read(size_t count = 0, size_t offset = 0) const;

    /**
     * @brief read data from the buffer into a array
     * @param data pointer where data is read to
     * @param count number of elements to read
     * @param offset offset where to start reading
     */
    void read( T* data, size_t count, size_t offset = 0) const;

    // element access
    T& operator[](std::size_t idx); //!< access an element in the buffer, "map" and "enableWrite" needs to be true
    const T& operator[](std::size_t idx) const; //!< access an element in the buffer readonly, "map" and needs to be true

    // iterators
    typedef T* iterator;
    typedef const T* const_iterator;

    iterator begin(); //!< get an iterator to the beginning of the buffer, "map" and "enableWrite" needs to be true
    iterator end(); //!< get an iterator to the end of the buffer, "map" and "enableWrite" needs to be true
    const_iterator cbegin() const; //!< get a const iterator to the beginning of the buffer, "map" needs to be true
    const_iterator cend() const; //!< get a const iterator to the end of the buffer, "map" and needs to be true

    size_t size() const { return m_size;} //!< returns the size of the buffer

private:

    GLbitfield storageFlags(); //!< generates bufferStorage flags based on template parameters
    GLbitfield mapAccesFlags(); //!< generates flags for MapBufferRange based on template parameters

    size_t m_size; //!< size of the buffer
    T* m_mapped_data; //!< pointer to the mapped data
};

//-------------------------------------------------------------------
// definitions of template functions of the Buffer class

template <typename T, bool map, bool enableWrite>
Buffer<T, map, enableWrite>::Buffer(size_t size) : BufferBase(), m_size(size), m_mapped_data(nullptr)
{
    glNamedBufferStorage(*this, size*sizeof(T), nullptr, storageFlags());
    if(map)
    {
        m_mapped_data = reinterpret_cast<T*>( glMapNamedBufferRange(*this, 0, m_size*sizeof(T), mapAccesFlags()));
        assert_true(m_mapped_data, "Buffer", "Mapping Buffer failed");
    }
}

template <typename T, bool map, bool enableWrite>
Buffer<T, map, enableWrite>::Buffer(std::vector<T> data) : BufferBase(), m_size(data.size), m_mapped_data(nullptr)
{
    glNamedBufferStorage(*this, data.size() * sizeof(T), data.data(), storageFlags());
    if(map)
    {
        m_mapped_data = reinterpret_cast<T*>( glMapNamedBufferRange(*this, 0, m_size*sizeof(T), mapAccesFlags()));
        assert_true(m_mapped_data, "Buffer", "Mapping Buffer failed");
    }
}

template <typename T, bool map, bool enableWrite>
Buffer<T, map, enableWrite>::Buffer(T* data, size_t count) : BufferBase(), m_size(count), m_mapped_data(nullptr)
{
    glNamedBufferStorage(*this, count * sizeof(T), data, storageFlags());
    if(map)
    {
        m_mapped_data = reinterpret_cast<T*>( glMapNamedBufferRange(*this, 0, m_size*sizeof(T), mapAccesFlags()));
        assert_true(m_mapped_data, "Buffer", "Mapping Buffer failed");
    }
}

template <typename T, bool map, bool enableWrite>
Buffer<T, map, enableWrite>::~Buffer()
{
    glUnmapNamedBuffer(*this);
}

template <typename T, bool enableWrite, bool map>
template <bool isWrite, bool isMapped>
void Buffer<T, enableWrite, map>::copy(const Buffer<T,isWrite,isMapped>& source, size_t count, size_t sourceOffset, size_t destOffset)
{
    if(count == 0)
        count = source.size()-sourceOffset;
    glCopyNamedBufferSubData(source,*this,sourceOffset,destOffset, count * sizeof(T));
}

template <typename T, bool enableWrite, bool map>
Buffer<T, enableWrite, map> Buffer<T, enableWrite, map>::clone()
{
    Buffer clone(this->size());
    glCopyNamedBufferSubData(*this,clone,0,0,size());
    return clone;
}

template <typename T, bool map, bool enableWrite>
void Buffer<T, map, enableWrite>::write(std::vector<T> data, size_t offset)
{
    static_assert(enableWrite,"To use write to the buffer you need to set the template parameter \"enableWrite\" to true.");
    if(map)
    {
        std::copy(data.begin(), data.end(), m_mapped_data+offset);
    } else
    {
        glNamedBufferSubData(*this, offset* sizeof(T), data.size()* sizeof(T), data.data());
    }
}

template <typename T, bool map, bool enableWrite>
void Buffer<T, map, enableWrite>::write(T* data, size_t count, size_t offset)
{
    static_assert(enableWrite,"To use write to the buffer you need to set the template parameter \"enableWrite\" to true.");
    if(map)
    {
        std::copy(data, data+count, m_mapped_data+offset);
    } else
    {
        glNamedBufferSubData(*this, offset* sizeof(T), count* sizeof(T), data);
    }
}

template <typename T, bool map, bool enableWrite>
std::vector<T> Buffer<T, map, enableWrite>::read(size_t count, size_t offset) const
{
    std::vector<T> data(count);
    if(map)
    {
        std::copy(m_mapped_data+offset, m_mapped_data+offset+count, data.begin());
    } else
    {
        glGetNamedBufferSubData(*this,offset*sizeof(T), count* sizeof(T), data.data());
    }
    return data;
}

template <typename T, bool map, bool enableWrite>
void Buffer<T, map, enableWrite>::read(T* data, size_t count, size_t offset) const
{
    if(map)
    {
        std::copy(m_mapped_data+offset, m_mapped_data+offset+count, data);
    } else
    {
        glGetNamedBufferSubData(*this,offset*sizeof(T), count* sizeof(T), data());
    }
}

template <typename T, bool enableWrite, bool map>
T& Buffer<T, enableWrite, map>::operator[](std::size_t idx)
{
    static_assert(map,"To use array access operators on a buffer you must set the template parameter \"map\" to true.");
    static_assert(enableWrite,"To use write to the buffer you need to set the template parameter \"enableWrite\" to true.");
    return *(m_mapped_data+idx);
}

template <typename T, bool enableWrite, bool map>
const T& Buffer<T, enableWrite, map>::operator[](std::size_t idx) const
{
    static_assert(map,"To use array access operators on a buffer you must set it to mapped.");
    return *(m_mapped_data+idx);
}

template <typename T, bool enableWrite, bool map>
typename Buffer<T, enableWrite, map>::iterator Buffer<T, enableWrite, map>::begin()
{
    static_assert(map,"To use array access operators on a buffer you must set the template parameter \"map\" to true.");
    static_assert(enableWrite,"To use write to the buffer you need to set the template parameter \"enableWrite\" to true.");
    return m_mapped_data;
}

template <typename T, bool enableWrite, bool map>
typename Buffer<T, enableWrite, map>::iterator Buffer<T, enableWrite, map>::end()
{
    static_assert(map,"To use array access operators on a buffer you must set the template parameter \"map\" to true.");
    static_assert(enableWrite,"To use write to the buffer you need to set the template parameter \"enableWrite\" to true.");
    return m_mapped_data + size();
}

template <typename T, bool enableWrite, bool map>
typename Buffer<T, enableWrite, map>::const_iterator Buffer<T, enableWrite, map>::cbegin() const
{
    static_assert(map,"To use array access operators on a buffer you must set the template parameter \"map\" to true.");
    return m_mapped_data;
}

template <typename T, bool enableWrite, bool map>
typename Buffer<T, enableWrite, map>::const_iterator Buffer<T, enableWrite, map>::cend() const
{
    static_assert(map,"To use array access operators on a buffer you must set the template parameter \"map\" to true.");
    return m_mapped_data + size();
}

template <typename T, bool map, bool enableWrite>
GLbitfield Buffer<T, map, enableWrite>::storageFlags()
{
    GLbitfield flags = 0;

    if(map)
        flags |= GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    if(enableWrite)
        flags |= GL_DYNAMIC_STORAGE_BIT;
    if(map && enableWrite)
        flags |= GL_MAP_WRITE_BIT;

    return flags;
}

template <typename T, bool map, bool enableWrite>
GLbitfield Buffer<T, map, enableWrite>::mapAccesFlags()
{
    GLbitfield flags = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;

    if(enableWrite)
        flags |= GL_MAP_WRITE_BIT;

    return flags;
}

}}