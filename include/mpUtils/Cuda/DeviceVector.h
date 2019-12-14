/*
 * mpUtils
 * DeviceVector.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the DeviceVector class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_DEVICEVECTOR_H
#define MPUTILS_DEVICEVECTOR_H

// includes
//--------------------
#include <initializer_list>
#include <type_traits>
#include <iterator>
#include <memory>
#include "clionCudaHelper.h"
#include "cudaUtils.h"
#include "mpUtils/Misc/templateUtils.h"
#include "VectorReference.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
/**
 * @brief references data in gpu memory from the cpu. Will update value in its destructor if it was changed. This should always only be a tempoary
 */
template<typename T>
class cudaAwareReference
{
public:
    explicit cudaAwareReference(T* gpuAdress) : m_gpuAdress(gpuAdress) {assert_cuda(cudaMemcpy(&m_cpuCopy,m_gpuAdress,sizeof(T),cudaMemcpyDeviceToHost));}
    ~cudaAwareReference() {if(m_changed) assert_cuda(cudaMemcpy(m_gpuAdress,&m_cpuCopy,sizeof(T),cudaMemcpyHostToDevice)); }

    cudaAwareReference& operator=(T other )
    {
        m_changed = true;
        m_cpuCopy = other;
        return *this;
    }

    const cudaAwareReference& operator+=(T other )
    {
        m_changed = true;
        m_cpuCopy += other;
        return *this;
    }

    const cudaAwareReference& operator-=(T other )
    {
        m_changed = true;
        m_cpuCopy -= other;
        return *this;
    }

    const cudaAwareReference& operator*=(T other )
    {
        m_changed = true;
        m_cpuCopy *= other;
        return *this;
    }

    const cudaAwareReference& operator/=(T other )
    {
        m_changed = true;
        m_cpuCopy /= other;
        return *this;
    }

    const cudaAwareReference& operator%=(T other )
    {
        m_changed = true;
        m_cpuCopy %= other;
        return *this;
    }

    const cudaAwareReference& operator^=(T other )
    {
        m_changed = true;
        m_cpuCopy ^= other;
        return *this;
    }

    const cudaAwareReference& operator|=(T other )
    {
        m_changed = true;
        m_cpuCopy |= other;
        return *this;
    }

    const cudaAwareReference& operator&=(T other )
    {
        m_changed = true;
        m_cpuCopy &= other;
        return *this;
    }

    const cudaAwareReference& operator>>=(T other )
    {
        m_changed = true;
        m_cpuCopy >>= other;
        return *this;
    }

    const cudaAwareReference& operator<<=(T other )
    {
        m_changed = true;
        m_cpuCopy <<= other;
        return *this;
    }

    const cudaAwareReference& operator!=(T other )
    {
        m_changed = true;
        m_cpuCopy != other;
        return *this;
    }

    cudaAwareReference& operator++()
    {
        m_changed = true;
        m_cpuCopy++;
        return *this;
    }

    T operator++(int)
    {
        T copy = m_cpuCopy;
        m_changed = true;
        m_cpuCopy++;
        return copy;
    }

    cudaAwareReference& operator--()
    {
        m_changed = true;
        m_cpuCopy--;
        return *this;
    }

    T operator--(int)
    {
        T copy = m_cpuCopy;
        m_changed = true;
        m_cpuCopy--;
        return copy;
    }

    operator const T&() const
    {
        return m_cpuCopy;
    }

private:
    T* m_gpuAdress;
    bool m_changed{false};
    T m_cpuCopy;
};

//-------------------------------------------------------------------
/**
 * class DeviceVector
 *
 * Vector class, should feature the same methods as std vector and is used in the same way. Except memory lives in gpu global memory.
 * It does not provide any iterators to access internal data as those would also need to be cuda aware.
 * Some functions are missing from the original vector. Eg emplace, emplace back as well as move operations do not make sense as data needs to be
 * copied to the gpu anyways. Iterators are also not implemented as they would need to be cuda aware and would likely be slow. Feel free to implement them yourself ;).
 *
 * usage:
 * Use like std vector. Call make_vectorReference to use the vector data inside of a cuda kernel function.
 * Value initialization can be turned of when using resize(int) or DeviceVector(int), by passing false as the second argument.
 * assignFromDeviceMem() and insertFromDeviceMem() can be used to copy values that are already in the gpu memory into the vector.
 * Also see notes!
 *
 * notes:
 * Remember, that capacity changes will invalidate all pointer, references on internal data, as well as the vectorReference
 * Also remember, that all assignments of values call cudaMemcopy to actually assign the values. Consider copying to/from a PinnedVector
 * when changing a lot of data.
 * It is assumed that elements can be constructed / destructed on host and device.
 * Functions creating / destructing a single argument will do so in host code and then copy the data to the device
 * (eg pop_back downloads the object and then destructs it).
 * Functions that are provided with data to be copied into the vector will simply copy the raw data to gpu memory, without invoking a copy/move constructor or similar.
 * Functions that construct multiple elements will do so from device code (using a cuda kernel) (eg DeviceVector(count), DeviceVector(count,value) or resize() ).
 * To initialize data on the cpu set constructOnDevice to false.
 *
 */
template <typename T, bool constructOnDevice = true>
class DeviceVector
{
public:
    // constructors
    DeviceVector(); //!< construct empty device vector
    DeviceVector(int count, const T& value); //!< construct device vector and fill it with count elements of value
    explicit DeviceVector(int count, bool valueInitialize = true); //!< construct vector with count elements, valueInitialize determains if they are initialized/constructed or not (using value initialization)

    DeviceVector( const T* first, int count); //!< construct vector and upload count elements starting with first to the gpu

    // destructor
    ~DeviceVector();

    // copy and move
    DeviceVector(const DeviceVector& other);
    DeviceVector(DeviceVector&& other) noexcept;
    DeviceVector& operator=(DeviceVector other) noexcept ;

    // conversion
    DeviceVector(std::initializer_list<T> ilist); //!< construct with values from an initialization list
    DeviceVector& operator=(std::initializer_list<T> ilist); //!< assign from initializer list

    // assign functions
    void assign(int count, const T& value); //!< assign count values of value to the vector
    void assign(const T* first, int count); //!< upload count elements starting with first to the gpu
    void assign(std::initializer_list<T> ilist); //!< assign values from initializer list
    void assignFromDeviceMem(const T* first, int count); //!< copy count elements into the vector, first needs to point to global device memory

    // element access
    cudaAwareReference<T> at(int idx); //!< access element idx with bounds checking
    const T& at(int idx) const; //!< access element idx with bounds checking
    cudaAwareReference<T> operator[](int idx); //!< access element idx without bounds checking
    const T& operator[](int idx) const; //!< access element idx without bounds checking
    cudaAwareReference<T> front(); //!< access first element
    const T& front() const; //!< access first element
    cudaAwareReference<T> back(); //!< access last element
    const T& back() const; //!< access last element
    T* data(); //!< direct access to pointer (GPU memory!)
    const T* data() const; //!< direct access to pointer (GPU memory!)

    // interators
    // not supported (yet) as they would have to be cuda aware as well and upload to the gpu after every store which would not be very fast

    // capacity
    bool empty() const noexcept; //!< check if vector is empty
    int size() const; //!< returns number of elements in the container
    int max_size() const; //!< max number of elements that can be stored (in practice gpu ram is smaller anyways)
    void reserve(int new_cap); //!< increase the capacity if new_cap is bigger then current capacity (does not increase size!)
    int capacity() const noexcept ; //!< returns current capacity before vector is resized
    void shrink_to_fit(); //!< non binding request to reduce capacity down to size

    // modifiers
    void clear() noexcept; //!< erases all items from the container
    // insert should be implemented
    // erase should be implemented
    void push_back(const T& value); //!< copy construct element into the end of vector
    void pop_back(); //!< removes the last element from the vector
    void resize(int count, bool valueInitialize = true); //!< sets size of container to a size of count. If elements are added valueInitialize determains if they are initialized/constructed or not (using value initialization)
    void resize(int count, const T& value); //!< sets size of container to a size of count. If elements are added they are initialized from value

    friend DeviceVector swap(const DeviceVector& first, const DeviceVector& second) noexcept
    {
        using std::swap;
        swap(first.m_data,second.m_data);
        swap(first.m_size,second.m_size);
        swap(first.m_capacity,second.m_capacity);
    }

private:
    T* m_data;
    int m_size;
    int m_capacity;

    static constexpr bool m_kNeedsDestruction = ! std::is_trivial<T>::value;

    // allocation / deallocation
    T* allocate(int count); //!< allocate cuda device memory for count elements of T
    void deallocate(T* ptr); //!< deallocate ptr

    // data movement
    void copy(T* dst, const T* src, int count); //!< copy count elements of type T from src to dst on device memory
    void download(T* host, const T* dev, int count); //!< download data from gpu to cpu memory
    void upload(T* dev, const T* host, int count); //!< download data from gpu to cpu memory

    // destruction (call with device memory pointer)
    void destruct(T* p, int count); //!< destructs count elements starting with p
    void destruct(T* p); //!< destructs a single element

    // construction (call with device memory pointer)
    template<typename ... Args>
    void construct(T* p, int count, Args...args); //!< construct count elements of T starting at localtion p

    // vector helper functions
    void reallocateStorage(int newCap); //!< allocates new storage of size newCap and copys over all data that fits
};

//-------------------------------------------------------------------
// some helper functions

/**
 * @brief Creates a vector reference for use on the device from a device vector
 * @param vec the device vector
 * @returns the VectorReference which references the device vector
 */
template <typename T>
VectorReference<T> make_vectorReference(DeviceVector<T>& vec)
{
    return VectorReference<T>(vec.data(),vec.size());
}

/**
 * @brief Creates a vector reference to const for use on the device from a const device vector
 * @param vec the device vector
 * @returns the VectorReference which references the device vector
 */
template <typename T>
VectorReference<const T> make_vectorReference(const DeviceVector<T>& vec)
{
    return VectorReference<const T>(vec.data(),vec.size());
}

//-------------------------------------------------------------------
// template function definitions for the device vector

template <typename T, bool constructOnDevice>
T* DeviceVector<T,constructOnDevice>::allocate(int count)
{
    T* ptr;
    assert_cuda(cudaMalloc(&ptr, sizeof(T)*count));
    return ptr;
}

template <typename T, bool constructOnDevice>
void DeviceVector<T,constructOnDevice>::deallocate(T* ptr)
{
    assert_cuda(cudaFree(ptr));
}

template <typename T, bool constructOnDevice>
void DeviceVector<T,constructOnDevice>::copy(T* dst, const T* src, int count)
{
    assert_cuda(cudaMemcpy(dst,src,sizeof(T)*count,cudaMemcpyDeviceToDevice));
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::download(T* host, const T* dev, int count)
{
    assert_cuda(cudaMemcpy(host,dev,sizeof(T)*count,cudaMemcpyDeviceToHost));
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::upload(T* dev, const T* host, int count)
{
    assert_cuda(cudaMemcpy(dev,host,sizeof(T)*count,cudaMemcpyHostToDevice));
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::destruct(T* p, int count)
{
    // only destruct stuff if actually needed
    if(m_kNeedsDestruction)
    {
        //    if(constructOnDevice)
//    {
        // TODO: faster to deconstruct data on the device!
//    } else
        {
            T* temp = (T*)(::operator new(sizeof(T)*count));
            download(temp,p,count);

            for(int i=0; i < count; i++)
                temp[i].~T();

            ::operator delete((void*)(temp));
        }
    }
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::destruct(T* p)
{
    destruct(p,1);
}

template <typename T, bool constructOnDevice>
template <typename... Args>
void DeviceVector<T, constructOnDevice>::construct(T* p, int count, Args... args)
{
//    if(constructOnDevice)
//    {
            // TODO: faster to construct data on the device!
//    } else
    {
        T* temp = (T*)(::operator new(sizeof(T)*count));

        for(int i=0; i < count; i++)
            new(temp+i) T(std::forward<Args>(args)...);

        upload(p,temp,count);
        ::operator delete((void*)(temp));
    }
}

template <typename T, bool constructOnDevice>
void DeviceVector<T,constructOnDevice>::reallocateStorage(int newCap)
{
    T* newMemory = allocate(newCap);

    if(m_data)
        copy(newMemory,m_data,min(m_capacity,newCap));

    deallocate(m_data);
    m_data = newMemory;
    m_capacity = newCap;
}

template <typename T, bool constructOnDevice>
DeviceVector<T,constructOnDevice>::DeviceVector() : m_size(0), m_capacity(0), m_data(nullptr)
{
}

template <typename T, bool constructOnDevice>
DeviceVector<T, constructOnDevice>::DeviceVector(const T* first, int count) : m_size(count), m_capacity(count)
{
    m_data = allocate(count);
    upload(m_data,first,count);
}

template <typename T, bool constructOnDevice>
DeviceVector<T, constructOnDevice>::DeviceVector(int count, const T& value)
{
    m_data = allocate(count);
    construct(m_data, count, value);
    m_capacity = count;
    m_size = count;

}

template <typename T, bool constructOnDevice>
DeviceVector<T, constructOnDevice>::DeviceVector(int count, bool valueInitialize) : DeviceVector()
{
    m_data = allocate(count);
    if(valueInitialize)
        construct(m_data, count);
    m_capacity = count;
    m_size = count;
}

template <typename T, bool constructOnDevice>
DeviceVector<T,constructOnDevice>::~DeviceVector()
{
    if(size()>0)
        destruct(m_data,m_size);
    if(capacity()>0)
        deallocate(m_data);
}

template <typename T, bool constructOnDevice>
DeviceVector<T, constructOnDevice>::DeviceVector(const DeviceVector& other) : m_size(other.size()), m_capacity(other.size())
{
    m_data = allocate(other.size());
    copy(m_data,other.data(),other.size());
}

template <typename T, bool constructOnDevice>
DeviceVector<T, constructOnDevice>::DeviceVector(DeviceVector&& other) noexcept : DeviceVector()
{
    swap(*this,other);
}

template <typename T, bool constructOnDevice>
DeviceVector<T, constructOnDevice>& DeviceVector<T, constructOnDevice>::operator=(DeviceVector other) noexcept
{
    swap(*this,other);
    return *this;
}

template <typename T, bool constructOnDevice>
DeviceVector<T, constructOnDevice>::DeviceVector(std::initializer_list<T> ilist) : DeviceVector(ilist.begin(),ilist.size())
{
}

template <typename T, bool constructOnDevice>
DeviceVector<T, constructOnDevice>& DeviceVector<T, constructOnDevice>::operator=(std::initializer_list<T> ilist)
{
    assign(ilist);
    return *this;
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::assign(int count, const T& value)
{
    destruct(m_data, m_size);

    if(m_capacity < count)
    {
        // don't use reallocate() because that would copy data
        deallocate(m_data);
        m_data = allocate(count);
        m_capacity = count;
    }

    construct(m_data,count,value);
    m_size = count;
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::assign(const T* first, int count)
{
    destruct(m_data, m_size);

    if(m_capacity < count)
    {
        // don't use reallocate() because that would copy data
        deallocate(m_data);
        m_data = allocate(count);
        m_capacity = count;
    }

    copy(m_data,first,count);
    m_size = count;
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::assign(std::initializer_list<T> ilist)
{
    assign(ilist.begin,ilist.size());
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::assignFromDeviceMem(const T* first, int count)
{
    destruct(m_data, m_size);

    if(m_capacity < count)
    {
        // don't use reallocate() because that would copy data
        deallocate(m_data);
        m_data = allocate(count);
        m_capacity = count;
    }

    upload(m_data,first,count);
    m_size = count;
}

template <typename T, bool constructOnDevice>
bool DeviceVector<T, constructOnDevice>::empty() const noexcept
{
    return m_size==0;
}

template <typename T, bool constructOnDevice>
int DeviceVector<T, constructOnDevice>::size() const
{
    return m_size;
}

template <typename T, bool constructOnDevice>
int DeviceVector<T, constructOnDevice>::max_size() const
{
    return std::numeric_limits<int>::max();
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::reserve(int new_cap)
{
    if(new_cap > m_capacity)
        reallocateStorage(new_cap);
}

template <typename T, bool constructOnDevice>
int DeviceVector<T, constructOnDevice>::capacity() const noexcept
{
    return m_capacity;
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::shrink_to_fit()
{
    reallocateStorage(m_size);
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::push_back(const T& value)
{
    if(m_size+1 > m_capacity)
        reallocateStorage(m_capacity*2 +1);

    upload(m_data+m_size,&value,1);
    m_size++;
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::clear() noexcept
{
    destruct(m_data,m_size);
    m_size = 0;
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::pop_back()
{
    destruct(m_data+m_size-1);
    m_size--;
}

template <typename T, bool constructOnDevice>
cudaAwareReference<T> DeviceVector<T, constructOnDevice>::at(int idx)
{
    assert_critical(idx>0&&idx<m_size,"DeviceVector","Access out of bounds!");
    return (*this)[idx];
}

template <typename T, bool constructOnDevice>
const T& DeviceVector<T, constructOnDevice>::at(int idx) const
{
    assert_critical(idx>0&&idx<m_size,"DeviceVector","Access out of bounds!");
    return (*this)[idx];
}

template <typename T, bool constructOnDevice>
cudaAwareReference<T> DeviceVector<T, constructOnDevice>::operator[](int idx)
{
    return cudaAwareReference<T>(m_data+idx);
}

template <typename T, bool constructOnDevice>
const T& DeviceVector<T, constructOnDevice>::operator[](int idx) const
{
    T v;
    download(&v,m_data+idx,1);
    return v;
}

template <typename T, bool constructOnDevice>
cudaAwareReference<T> DeviceVector<T, constructOnDevice>::front()
{
    return (*this)[0];
}

template <typename T, bool constructOnDevice>
const T& DeviceVector<T, constructOnDevice>::front() const
{
    return (*this)[0];
}

template <typename T, bool constructOnDevice>
cudaAwareReference<T> DeviceVector<T, constructOnDevice>::back()
{
    return (*this)[m_size-1];
}

template <typename T, bool constructOnDevice>
const T& DeviceVector<T, constructOnDevice>::back() const
{
    return (*this)[m_size-1];
}

template <typename T, bool constructOnDevice>
T* DeviceVector<T, constructOnDevice>::data()
{
    return m_data;
}

template <typename T, bool constructOnDevice>
const T* DeviceVector<T, constructOnDevice>::data() const
{
    return m_data;
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::resize(int count, bool valueInitialize)
{

    if(m_capacity < count)
        reallocateStorage(count);

    if(m_size > count)
        destruct(m_data+count,m_size-count);

    if(m_size < count && valueInitialize)
        construct(m_data+m_size,count-m_size);

    m_size  = count;
}

template <typename T, bool constructOnDevice>
void DeviceVector<T, constructOnDevice>::resize(int count, const T& value)
{
    if(m_capacity < count)
        reallocateStorage(count);

    if(m_size > count)
        destruct(m_data+count,m_size-count);

    if(m_size < count)
        construct(m_data+m_size,count-m_size, value);

    m_size  = count;
}

}

#endif //MPUTILS_DEVICEVECTOR_H
