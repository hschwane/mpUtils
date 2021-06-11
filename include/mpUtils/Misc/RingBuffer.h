/*
 * mpUtils
 * RingBuffer.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the RingBuffer class, a circular buffer used similar to an stl container.
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_RINGBUFFER_H
#define MPUTILS_RINGBUFFER_H

// includes
//--------------------
#include <vector>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
/**
 * class RingBuffer
 *
 * usage:
 * T is a type, Container a container like vector or array that provides random access.
 * If Container allows resizing, the ringbuffer can be resized.
 * Call push to add elements to the END of the ring buffer, call pop to remove them at the beginning.
 *
 */
template <typename T, typename Container=std::vector<T>>
class RingBuffer
{
public:

    // types
    using container_type = Container;
    using value_type = typename Container::value_type;
    using size_type = typename Container::size_type;
    using reference = typename Container::reference;
    using const_reference = typename Container::const_reference;

    // constructors
    RingBuffer();
    RingBuffer(size_type n);
    RingBuffer(const RingBuffer& other);
    RingBuffer(RingBuffer&& other);

    explicit RingBuffer(const Container& cont);
    explicit RingBuffer(Container&& cont);

    // operator=
    RingBuffer& operator=(const RingBuffer& other);
    RingBuffer& operator=(RingBuffer&& other);

    // check size and capacity
    bool empty(); //! true, if the buffer is empty
    bool full(); //! true if the buffer is full, and adding another element will overwrite the oldest element
    size_type size(); //! returns number of elements stored in the buffer
    size_type capacity(); //! returns the maximum number of elements that could fit into the buffer

    // modify the buffer
    void resize(size_type newCapacity); //! change capacity of ringbuffer if underlying container allows it
    void clear(); //! discard all elements in the buffer

    void push(const value_type& value); //! adds "value" to the ringbuffer
    void push(value_type&& value); //! adds "value" to the ringbuffer

    template <class... Args>
    decltype(auto) emplace(Args&& ... args);

    void pop(); //! removes an element at the back of the buffer

    // access
    reference front();
    const_reference front() const;
    reference back();
    const_reference back() const;
    reference operator[](size_t id);
    const_reference operator[](size_t id) const;

private:
    container_type m_container;
    size_type m_insert;
    size_type m_read;
    size_type m_contentSize;
};

//-------------------------------------------------------------------
// template funciton definitions of the ringbuffer class

template <typename T, typename Container>
RingBuffer<T, Container>::RingBuffer() : m_container(), m_insert(0), m_read(0)
{
}

template <typename T, typename Container>
RingBuffer<T, Container>::RingBuffer(size_type n) : m_container(n), m_insert(0), m_read(0)
{
}

template <typename T, typename Container>
RingBuffer<T, Container>::RingBuffer(const RingBuffer& other)
    : m_container(other.m_container), m_insert(other.m_insert), m_read(other.m_read)
{
}

template <typename T, typename Container>
RingBuffer<T, Container>::RingBuffer(RingBuffer&& other)
{

}

template <typename T, typename Container>
RingBuffer<T, Container>::RingBuffer(const Container& cont)
{

}

template <typename T, typename Container>
RingBuffer<T, Container>::RingBuffer(Container&& cont)
{

}

template <typename T, typename Container>
typename RingBuffer<T, Container>::RingBuffer& RingBuffer<T, Container>::operator=(const RingBuffer& other)
{
    return <#initializer#>;
}

template <typename T, typename Container>
typename RingBuffer<T, Container>::RingBuffer& RingBuffer<T, Container>::operator=(RingBuffer&& other)
{
    return <#initializer#>;
}

template <typename T, typename Container>
bool RingBuffer<T, Container>::empty()
{
    return false;
}

template <typename T, typename Container>
bool RingBuffer<T, Container>::full()
{
    return false;
}

template <typename T, typename Container>
typename RingBuffer<T, Container>::size_type RingBuffer<T, Container>::size()
{
    return nullptr;
}

template <typename T, typename Container>
typename RingBuffer<T, Container>::size_type RingBuffer<T, Container>::capacity()
{
    return nullptr;
}

template <typename T, typename Container>
void RingBuffer<T, Container>::resize(size_type newCapacity)
{

}

template <typename T, typename Container>
void RingBuffer<T, Container>::clear()
{

}

template <typename T, typename Container>
void RingBuffer<T, Container>::push(const value_type& value)
{

}

template <typename T, typename Container>
void RingBuffer<T, Container>::push(value_type&& value)
{

}

template <typename T, typename Container>
template <class... Args>
decltype(auto) RingBuffer<T, Container>::emplace(Args&& ... args)
{
    return nullptr;
}

template <typename T, typename Container>
void RingBuffer<T, Container>::pop()
{

}

template <typename T, typename Container>
typename RingBuffer<T, Container>::reference RingBuffer<T, Container>::front()
{
    return nullptr;
}

template <typename T, typename Container>
typename RingBuffer<T, Container>::const_reference RingBuffer<T, Container>::front() const
{
    return nullptr;
}

template <typename T, typename Container>
typename RingBuffer<T, Container>::reference RingBuffer<T, Container>::back()
{
    return nullptr;
}

template <typename T, typename Container>
typename RingBuffer<T, Container>::const_reference RingBuffer<T, Container>::back() const
{
    return nullptr;
}

template <typename T, typename Container>
typename RingBuffer<T, Container>::reference RingBuffer<T, Container>::operator[](size_t id)
{
    return nullptr;
}

template <typename T, typename Container>
typename RingBuffer<T, Container>::const_reference RingBuffer<T, Container>::operator[](size_t id) const
{
    return nullptr;
}

}
#endif //MPUTILS_RINGBUFFER_H
