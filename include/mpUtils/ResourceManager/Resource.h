/*
 * mpUtils
 * Resource.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Resource class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_RESOURCE_H
#define MPUTILS_RESOURCE_H

// includes
//--------------------
#include <utility>
#include "RefcountingHelper.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

// forward declaration
//--------------------
template <typename T, typename S> class ResourceCache;
//--------------------

//-------------------------------------------------------------------
/**
 * class Resource
 *
 * holds a resource of type T manages by the resource manager
 *
 */
template <typename T>
class Resource
{
public:
    using ResourceType = T;
    using HandleType = unsigned int;

    Resource()
            : m_resource(nullptr), m_handle(0), m_refcount(nullptr)
    {
    }

    Resource(const Resource& other)
            : m_resource(other.m_resource), m_handle(other.m_handle), m_refcount(other.m_refcount)
    {
        if(m_refcount)
            m_refcount->signalConstruction(m_handle);
    }

    Resource(Resource&& other)
            : m_resource(other.m_resource), m_handle(other.m_handle), m_refcount(other.m_refcount)
    {
        // we dont have a default constructor, so we cannot swap, simply steal from the other object
        // and make sure it does not decrease the ref counter on construction
        other.m_refcount = nullptr;
    }

    //!< assign (move and copy)
    Resource& operator=(Resource other)
    {
        swap(*this, other);
        return *this;
    }

    ~Resource()
    {
        if(m_refcount)
            m_refcount->signalDestruction(m_handle);
    }

    const T* operator->() const { return m_resource; }
    const T& operator*() const { return *m_resource; }
    const T* get() const { return m_resource; }
    explicit operator bool() const noexcept {return (m_resource != nullptr);}

    void unload() {m_resource=nullptr; if(m_refcount)m_refcount->signalDestruction(m_handle); m_refcount= nullptr; }

    //!< swap for copy swap
    friend void swap(Resource& first, Resource& second)
    {
        using std::swap;
        swap(first.m_resource, second.m_resource);
        swap(first.m_handle, second.m_handle);
        swap(first.m_refcount, second.m_refcount);
    }

private:

    //!< construct new resource (used by the manager)
    Resource(const T* pointerToResource, HandleType handle, RefcountingHelper* refcount)
            : m_resource(pointerToResource), m_handle(handle), m_refcount(refcount)
    {
        if(m_refcount)
            m_refcount->signalConstruction(m_handle);
    }
    template<typename U, typename V> friend class ResourceCache;

    const T* m_resource; //!< pointer to the resource
    HandleType m_handle; //!< handle to the resource in the manager
    RefcountingHelper* m_refcount; //!< rhe resource manager used to create the resource
};

}
// include forward declared classes
#include "mpUtils/ResourceManager/ResourceCache.h"

#endif //MPUTILS_RESOURCE_H
