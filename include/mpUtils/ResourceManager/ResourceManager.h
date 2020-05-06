/*
 * mpUtils
 * ResourceManager.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the ResourceManager class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_RESOURCEMANAGER_H
#define MPUTILS_RESOURCEMANAGER_H

// includes
//--------------------
#include "ResourceCache.h"
#include "mpUtils/Misc/templateUtils.h"
#include "mpUtils/external/threadPool/ThreadPool.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------



//-------------------------------------------------------------------
/**
 * class ResourceManager
 *
 * Resource manager packs together multiple resource caches for the management of resources of different types.
 *
 */
template <typename ... CacheT>
class ResourceManager
{
public:
    using resourceTypes = std::tuple<typename CacheT::ResourceType ...> ;

    /**
     * @brief one of those needs to be passed to the constructor for each used resource
     */
    template <typename T, typename PreloadDataT>
    struct cacheCreationData
    {
        std::function<std::unique_ptr<PreloadDataT>(std::string)> asyncPreloadFunc; //!< function for asynchronous preloading
        std::function<std::unique_ptr<T>(std::unique_ptr<PreloadDataT>)>  syncLoadFunc; //!< function to finish loading synchronously
        std::string workingDir; //!< working directory for this kind of resources
        std::unique_ptr<T> defaultResource; //!< default resource, used if resource is missing
    };

    explicit ResourceManager( cacheCreationData<typename CacheT::ResourceType, typename CacheT::PreloadType> ... caches);

    template <typename T> void preload(const std::string& path); //!< preloads a resource of type T with name path
    template <typename T> Resource<T> load(const std::string& path); //!< loads a resource of type T with name path

    template <typename T> bool isReady(const std::string& path); //!< check if resource is ready for use
    template <typename T> bool isPreloaded(const std::string& path); //!< check if resource is done preloading

    bool loadOne(); //!< syncronously loads one resource that finished preloading, use for loading screens etc. return false if there is nothing to load anymore

    template <typename T> void forceReload(const std::string& path); //!< force reload a specific resource  race conditions might occur if resources simultaneously accessed in another thread
    template <typename T> void tryRelease(const std::string& path); //! releases resource, if it is not used anymore
    void forceReloadAll(); //!< reloads all resources
    void tryReleaseAll(); //!< removes all resources that are not used anymore

    int numLoaded(); //!< total number of loaded resources

    template <typename T> auto& get(); //!< returns reference to the resource cache for resources of type T

    int getNumThreads(); //!< number of threads that are used for background loading
    void setNumThreads(int threads); //!< number of threads that are used for background loading

private:
    using preloadTypes = std::tuple<typename CacheT::PreloadType ...>;
    std::tuple<std::unique_ptr<CacheT>...> m_caches;

    ThreadPool m_threadPool;
};

// template function definition
//-------------------------------------------------------------------

template <typename... CacheT>
ResourceManager<CacheT...>::ResourceManager( cacheCreationData<typename CacheT::ResourceType, typename CacheT::PreloadType> ... caches)
    : m_threadPool(2), m_caches( std::make_unique<CacheT>( caches.asyncPreloadFunc,
                                                                caches.syncLoadFunc,
                                                                caches.workingDir,
                                                                [this](std::function<void()> f){this->m_threadPool.enqueue(f);},
                                                                std::move(caches.defaultResource)     ) ... )
{
}

template <typename... CacheT>
template <typename T>
Resource<T> ResourceManager<CacheT...>::load(const std::string& path)
{
    static_assert(has_type_v<T,resourceTypes>,"Resource manager has no cache to deal with resources of that type.");

    // get preload type to the given resource type
    using PreloadType = std::tuple_element_t<index_of_v<T,resourceTypes>,preloadTypes>;
    return std::get<std::unique_ptr<ResourceCache<T,PreloadType>>>(m_caches)->load(path);
}

template <typename... CacheT>
template <typename T>
void ResourceManager<CacheT...>::preload(const std::string& path)
{
    static_assert(has_type_v<T,resourceTypes>,"Resource manager has no cache to deal with resources of that type.");

    // get preload type to the given resource type
    using PreloadType = std::tuple_element_t<index_of_v<T,resourceTypes>,preloadTypes>;
    std::get<std::unique_ptr<ResourceCache<T,PreloadType>>>(m_caches)->preload(path);
}

template <typename... CacheT>
template <typename T>
bool ResourceManager<CacheT...>::isReady(const std::string& path)
{
    static_assert(has_type_v<T,resourceTypes>,"Resource manager has no cache to deal with resources of that type.");

    // get preload type to the given resource type
    using PreloadType = std::tuple_element_t<index_of_v<T,resourceTypes>,preloadTypes>;
    return std::get<std::unique_ptr<ResourceCache<T,PreloadType>>>(m_caches)->isReady(path);
}

template <typename... CacheT>
template <typename T>
bool ResourceManager<CacheT...>::isPreloaded(const std::string& path)
{
    static_assert(has_type_v<T,resourceTypes>,"Resource manager has no cache to deal with resources of that type.");

    // get preload type to the given resource type
    using PreloadType = std::tuple_element_t<index_of_v<T,resourceTypes>,preloadTypes>;
    return std::get<std::unique_ptr<ResourceCache<T,PreloadType>>>(m_caches)->isPreloaded(path);
}

template <typename... CacheT>
template <typename T>
void ResourceManager<CacheT...>::forceReload(const std::string& path)
{
    // get preload type to the given resource type
    using PreloadType = std::tuple_element_t<index_of_v<T,resourceTypes>,preloadTypes>;
    return std::get<std::unique_ptr<ResourceCache<T,PreloadType>>>(m_caches)->forceReload(path);
}

template <typename... CacheT>
template <typename T>
void ResourceManager<CacheT...>::tryRelease(const std::string& path)
{
    // get preload type to the given resource type
    using PreloadType = std::tuple_element_t<index_of_v<T,resourceTypes>,preloadTypes>;
    return std::get<std::unique_ptr<ResourceCache<T,PreloadType>>>(m_caches)->tryRelease(path);
}

template <typename... CacheT>
int ResourceManager<CacheT...>::getNumThreads()
{
    return m_threadPool.getPoolSize();
}

template <typename... CacheT>
void ResourceManager<CacheT...>::setNumThreads(int threads)
{
    m_threadPool.setPoolSize(threads);
}

template <typename... CacheT>
bool ResourceManager<CacheT...>::loadOne()
{
    return (std::get<CacheT>(m_caches).loadOne() | ...);
}

template <typename... CacheT>
void ResourceManager<CacheT...>::forceReloadAll()
{
    int t[] = {0, ((void)( std::get<CacheT>(m_caches)->forceReloadAll() ),1)...};
    (void)t[0]; // silence compiler warning about t being unused
}

template <typename... CacheT>
void ResourceManager<CacheT...>::tryReleaseAll()
{
    int t[] = {0, ((void)( std::get<CacheT>(m_caches)->tryReleaseAll() ),1)...};
    (void)t[0]; // silence compiler warning about t being unused
}

template <typename... CacheT>
int ResourceManager<CacheT...>::numLoaded()
{
    return (std::get<CacheT>(m_caches).numLoaded() + ...);
}

template <typename... CacheT>
template <typename T>
auto& ResourceManager<CacheT...>::get()
{
    static_assert(has_type_v<T,resourceTypes>,"Resource manager has no cache to deal with resources of that type.");

    // get preload type to the given resource type
    using PreloadType = std::tuple_element_t<index_of_v<T,resourceTypes>,preloadTypes>;
    return *(std::get<std::unique_ptr<ResourceCache<T,PreloadType>>>(m_caches));
}

}
#endif //MPUTILS_RESOURCEMANAGER_H
