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

#ifndef MPUTILS_RESOURCECACHE_H
#define MPUTILS_RESOURCECACHE_H

// includes
//--------------------
#include <unordered_map>
#include <string>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include "mpUtils/IO/readData.h"
#include "mpUtils/Log/Log.h"
#include "mpUtils/Misc/copyMoveWrapper.h"
#include "mpUtils/Misc/timeUtils.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

/**
 * @brief different states a resource can go through during loading
 */
enum class ResourceState
{
    none,       //!< handle was created, however resource has yet to be loaded
    preloading, //!< resource is currently preloading in the worker thread
    preloaded,  //!< resource finished preloading and is awaiting to be loaded
    preloadFailed,     //!< preloading failed, resource will be swapped with a default resource
    loading,     //!< resource is currently in the final syncronus loading step
    failed,     //!< loading failed, default resource will be used
    ready,       //!< resource is in the cache and ready to use
    defaulted   //!< default resource is used instead of this resource, as an error occured while loading
};

class ReloadMode
{
protected:
    static thread_local bool enabled;
};

//-------------------------------------------------------------------
/**
 * class ResourceCache
 *
 * Loads and manages resources of type T.
 * provides reference counting, duplication avoidance, filesystem abstraction, asynchronously loading and caching
 *
 * workDir is prepended to all path identifiers passed to the cache.
 * A preloadAsync and loadSync function must be provided. preloadAsync will be called with an std::string of data
 * (text/binary depending on the opened file) and is expected to produce a unique pointer to an object of type PreloadedDataT.
 * It might be called asynchronously in a worker thread.
 * loadSync will be called with the PreloadedData object and is expected to produce a unique pointer of type T.
 * It will always be run in the thread calling the load function.
 * startTask is expected to add the passed std::function to the used threadpool for execution.
 * The default resource will be loaded whenever a resource file could not be found
 *
 */
template <typename T, typename PreloadDataT>
class ResourceCache : private ReloadMode
{
public:
    using ResourceType = T;
    using PreloadType = PreloadDataT;
    using HandleType = unsigned int;

    ResourceCache(std::function<std::unique_ptr<PreloadDataT>(std::string)> preloadAsync,
            std::function<std::unique_ptr<T>(std::unique_ptr<PreloadDataT>)> loadSync,
            std::string workDir, std::function<void(std::function<void()>)> startTask,
            std::unique_ptr<T> defaultResource, std::string debugName)
            : m_asyncPreload(std::move(preloadAsync)), m_syncFinishLoad(std::move(loadSync)),
            m_startTask(std::move(startTask)), m_workDir(std::move(workDir)),
            m_defaultResource(std::move(defaultResource)), m_debugName(std::move(debugName))
    {
    }

    void setAddTaskFunc(std::function<void(std::function<void()>)> startTask); //!< change the add task function

    void preload(const std::string& path); //!< start preloading a resource
    std::shared_ptr<T> load(const std::string& path); //!< block until loading is finished

    bool isReady(const std::string& path); //!< check if resource is ready for use
    bool isPreloaded(const std::string& path); //!< check if resource is done preloading

    bool loadOne(); //!< syncronously loads one resource that finished preloading, use for loading screens etc. return false if there is nothing to load anymore

    void forceReloadAll(); //!< force reload on all resources race conditions might occur if resource is simultaneously accessed in another thread
    void forceReload(const std::string& path); //!< force reload a specific resource  race conditions might occur if resources simultaneously accessed in another thread
    void tryReleaseAll(); //!< removes all resources that are not used anymore
    void tryRelease(const std::string& path); //! releases resource, if it is not used anymore

    // for debugging and analysis:
    int numLoaded(); //!< returns number of loaded resources
    HandleType getHandle(const std::string& path); //!< returns internal resource handle (for debugging)
    std::tuple<const T*,const PreloadDataT*,int,ResourceState> getResourceInfo(HandleType h); //!< get information about resource h. Returns memory addresss, preload data adress, refcount and state. (for debugging)
    void doForEachResource(std::function<void(const std::string&, HandleType h)> f); //!< calls f for every currently loaded resource (for debugging)
    const std::string& getWorkDir() {return m_workDir;} //!< returns the working directory
    std::string& getDebugName() {return m_debugName;} //!< returns the name shown in the debugger

private:
    HandleType getResourceHandle(const std::string& path); //!< get a handle to the the path
    void doPreload(const std::string& path, HandleType handle); //!< function handles load from file, calling m_asyncPreload and creating the object
    void doReload(const std::string& path, HandleType handle); //!< synchronously reloads a resource into the same memory address as it was before

    std::string m_workDir; //!< working directory of the loader, will be prepended to all filenames
    std::string m_debugName; //!< name of this chache used for debugging and imgui

    std::function<void(std::function<void()>)> m_startTask; //!< forward a task to the used tasking system
    std::function<std::unique_ptr<PreloadDataT>(std::string data)> m_asyncPreload;    //!< executes part of loading that can be done in any thread, string contains binary or text data
    std::function<std::unique_ptr<T>(std::unique_ptr<PreloadDataT>)> m_syncFinishLoad;   //!< will be executed in the thread that called load()

    std::unordered_map<std::string, HandleType> m_resourceHandles; //!< map resource names to handles

    struct ResourceEntry
    {
        ResourceEntry() = default;
        std::shared_ptr<T> resource{nullptr};
        std::unique_ptr<PreloadDataT> preloadData{nullptr};
        CopyMoveAtomic<ResourceState> state{ResourceState::none};
    };
    std::vector<ResourceEntry> m_resources; //!< actual resources
    std::vector<HandleType> m_freeHandles; //!< free handles will go here for reuse

    std::shared_timed_mutex m_rhmtx; //!< mutex to lock the resource handle map
    std::shared_timed_mutex m_rmtx; //!< mutex for the resource vector
    std::mutex m_freeHandlesMtx; //!< mutex for free handles buffer
    std::shared_timed_mutex m_reloadAllLock; //!< allow only one reload all operartion at a time

    std::shared_ptr<T> m_defaultResource; //!< this will be used whenever a resource is missing

    template< typename A = T, typename std::enable_if<std::is_copy_constructible<A>::value,int>::type =0> std::shared_ptr<T> handleDefaultResource(HandleType h)
    {
        if(! m_resources[h].resource)
            m_resources[h].resource = std::make_shared<T>(*m_defaultResource);
        else
            *(m_resources[h].resource) = *m_defaultResource;
        m_resources[h].state = ResourceState::defaulted;
        return m_resources[h].resource;
    }
    template< typename A = T, typename std::enable_if< !std::is_copy_constructible<A>::value,int>::type =0> std::shared_ptr<T> handleDefaultResource(HandleType h)
    {
        m_resources[h].state = ResourceState::failed;
        return m_defaultResource;
    }
};

// template function definition
//-------------------------------------------------------------------
template <typename T, typename PreloadDataT>
void ResourceCache<T, PreloadDataT>::preload(const std::string& path)
{
    HandleType h = getResourceHandle(path);

    std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);
    if(m_resources[h].state == ResourceState::none)
    {
        sharedLck.unlock();
        m_startTask(std::bind(&ResourceCache::doPreload,this, path, h));
    }
}

template <typename T, typename PreloadDataT>
std::shared_ptr<T> ResourceCache<T, PreloadDataT>::load(const std::string& path)
{
    HandleType h = getResourceHandle(path);

    std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);

    // quick path in case resource is ready
    if(m_resources[h].state == ResourceState::ready || m_resources[h].state == ResourceState::defaulted)
    {
        if(ReloadMode::enabled)
            forceReload(path);

        return m_resources[h].resource;
    }

    if(m_resources[h].state == ResourceState::none)
    {
        // execute preloading step syncronously
        sharedLck.unlock();
        doPreload(path,h);
        sharedLck.lock();
    }
    else if(m_resources[h].state == ResourceState::preloading)
    {
        // wait for preloading to finish
        while(m_resources[h].state == ResourceState::preloading)
        {
            sharedLck.unlock();
            mpu::yield();
            sharedLck.lock();
        }
    }

    ResourceState expected = ResourceState::preloaded;
    bool failed=false;
    if(m_resources[h].state.compare_exchange_strong(expected,ResourceState::loading))
    {
        std::unique_ptr<PreloadDataT> pd = std::move(m_resources[h].preloadData);
        sharedLck.unlock();
        try
        {
            std::unique_ptr<T> r = m_syncFinishLoad(std::move(pd));
            sharedLck.lock();
            m_resources[h].resource = std::move(r);
            m_resources[h].state = ResourceState::ready;
        } catch(const std::exception& e)
        {
            logERROR("ResourceCache") << "Error loading resource " << path << ". Exception: " << e.what();
            if(!sharedLck.owns_lock())
                sharedLck.lock();
            failed = true;
        }
    } else if(expected == ResourceState::loading)
    {
        while(m_resources[h].state == ResourceState::loading)
        {
            sharedLck.unlock();
            std::this_thread::yield();
            sharedLck.lock();
        }
    }

    expected = ResourceState::preloadFailed;
    if(m_resources[h].state.compare_exchange_strong(expected,ResourceState::loading))
    {
        failed = true;
    }

    if(failed || m_resources[h].state == ResourceState::failed)
    {
        // resource loading failed, output the default resource instead
        return handleDefaultResource(h);
    }

    assert_true(m_resources[h].state == ResourceState::ready || m_resources[h].state == ResourceState::defaulted,
            "ResourceManager", "Resource is not ready after loading.");
    return m_resources[h].resource;
}

template <typename T, typename PreloadDataT>
void ResourceCache<T, PreloadDataT>::doPreload(const std::string& path, HandleType handle)
{
    ResourceState expected = ResourceState::none;
    if(!m_resources[handle].state.compare_exchange_strong(expected,ResourceState::preloading))
        return;

    try
    {
        std::string data = readFile(m_workDir + path);
        auto pd = m_asyncPreload(std::move(data));
        std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);
        m_resources[handle].preloadData = std::move(pd);
        m_resources[handle].state = ResourceState::preloaded;
    } catch(const std::exception& e)
    {
        logERROR("ResourceCache") << "Error preloading resource " << path << ". Exception: " << e.what();
        std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);
        m_resources[handle].state = ResourceState::preloadFailed;
    }
}

template <typename T, typename PreloadDataT>
void ResourceCache<T, PreloadDataT>::setAddTaskFunc(std::function<void(std::function<void()>)> startTask)
{
    m_startTask = std::move(startTask);
}

template <typename T, typename PreloadDataT>
typename ResourceCache<T,PreloadDataT>::HandleType ResourceCache<T,PreloadDataT>::getResourceHandle(const std::string& path)
{
    HandleType h;
    std::shared_lock<std::shared_timed_mutex> sharedLckRH(m_rhmtx);
    auto it = m_resourceHandles.find(path);
    if(it == m_resourceHandles.end())
    {
        sharedLckRH.unlock();

        std::unique_lock<std::mutex> lck(m_freeHandlesMtx);
        if(m_freeHandles.empty())
        {
            lck.unlock();
            std::unique_lock<std::shared_timed_mutex> lckRH(m_rhmtx);
            std::unique_lock<std::shared_timed_mutex> lckR(m_rmtx);
            h = m_resources.size();
            m_resources.emplace_back();
            m_resourceHandles.emplace(path,h);
            return h;
        } else
        {
            // use free handle from stack
            h = m_freeHandles.back();
            m_freeHandles.pop_back();
            return h;
        }
    } else
    {
        h = it->second;
    }
    return h;
}

template <typename T, typename PreloadDataT>
int ResourceCache<T, PreloadDataT>::numLoaded()
{
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);
    return m_resourceHandles.size();
}

template <typename T, typename PreloadDataT>
void ResourceCache<T, PreloadDataT>::doReload(const std::string& path, ResourceCache::HandleType h)
{
    bool failed = false;
    std::unique_ptr<T> r;
    try
    {
        r = m_syncFinishLoad( m_asyncPreload( readFile(m_workDir + path)));
    } catch(const std::exception& e)
    {
        logERROR("ResourceCache") << "Error reloading resource " << path << ". Exception: " << e.what();
        failed = true;
    }

    if(failed)
    {
        handleDefaultResource(h);
    } else
    {
        m_resources[h].state = ResourceState::loading;
        *(m_resources[h].resource) = std::move(*r);
        m_resources[h].state = ResourceState::ready;
    }
}

template <typename T, typename PreloadDataT>
void ResourceCache<T, PreloadDataT>::forceReloadAll()
{
    logINFO("ResourceManager") << m_debugName << " reloading everything.";
    std::unique_lock<std::shared_timed_mutex> lck(m_reloadAllLock);

    // make sure all dependencies are also reloaded
    bool prevReloadMode = ReloadMode::enabled;
    ReloadMode::enabled = true;

    // to use multiple threads
    std::shared_lock<std::shared_timed_mutex> sharedLckRH(m_rhmtx);
    for(auto& handle : m_resourceHandles)
    {
        HandleType h = handle.second;
        std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);
        if(m_resources[h].state == ResourceState::ready || m_resources[h].state == ResourceState::defaulted)
        {
            doReload(handle.first,h);
        }
    }

    // reset previous reload mode
    ReloadMode::enabled = prevReloadMode;

}

template <typename T, typename PreloadDataT>
void ResourceCache<T, PreloadDataT>::forceReload(const std::string& path)
{
    std::shared_lock<std::shared_timed_mutex> lck(m_reloadAllLock); // make sure nobody is reloading everything right now
    logINFO("ResourceManager") << "Forced to reload " + path;
    HandleType h = getResourceHandle(path);

    std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);
    if(!(m_resources[h].state == ResourceState::ready || m_resources[h].state == ResourceState::defaulted))
    {
        logDEBUG("ResourceManager") << "Force reload called on " << path << " which is not loaded";
    }

    // make sure all dependencies are also reloaded
    bool prevReloadMode = ReloadMode::enabled;
    ReloadMode::enabled = true;

    // reload to the same memory position
    doReload(path,h);

    // reset previous reload mode
    ReloadMode::enabled = prevReloadMode;
}

template <typename T, typename PreloadDataT>
void ResourceCache<T, PreloadDataT>::tryReleaseAll()
{
    logDEBUG("ResourceManager") << "Releasing all unused resources.";
    std::unique_lock<std::shared_timed_mutex> lckRH(m_rhmtx);
    std::unique_lock<std::shared_timed_mutex> lckR(m_rmtx);
    for(auto it = m_resourceHandles.cbegin(); it!=m_resourceHandles.cend(); )
    {
        HandleType h = it->second;
        if(m_resources[h].resource.use_count() == 1 && (m_resources[h].state == ResourceState::defaulted
            ||  m_resources[h].state == ResourceState::failed || m_resources[h].state == ResourceState::ready))
        {
            m_resources[h].state = ResourceState::none;
            m_resources[h].resource = nullptr;
            m_resources[h].preloadData = nullptr;
            lckR.unlock();
            it = m_resourceHandles.erase(it);
            lckRH.unlock();

            std::unique_lock<std::mutex> lckFreeHandles(m_freeHandlesMtx);
            m_freeHandles.push_back(h);
            lckFreeHandles.unlock();
            lckRH.lock();
            lckR.lock();
        } else
        {
            it++;
        }
    }
}

template <typename T, typename PreloadDataT>
void ResourceCache<T, PreloadDataT>::tryRelease(const std::string& path)
{
    logDEBUG("ResourceManager") << "Trying to release " + path;
    HandleType h = getResourceHandle(path);

    std::unique_lock<std::shared_timed_mutex> lckRH(m_rhmtx);
    std::unique_lock<std::shared_timed_mutex> lckR(m_rmtx);
    if(m_resources[h].resource.use_count() == 1 && (m_resources[h].state == ResourceState::defaulted
        ||  m_resources[h].state == ResourceState::failed || m_resources[h].state == ResourceState::ready))
    {
        m_resources[h].state = ResourceState::none;
        m_resources[h].resource = nullptr;
        m_resources[h].preloadData = nullptr;
        lckR.unlock();
        m_resourceHandles.erase(path);
        lckRH.unlock();

        std::unique_lock<std::mutex> lckFreeHandles(m_freeHandlesMtx);
        m_freeHandles.push_back(h);
    }
}

template <typename T, typename PreloadDataT>
typename ResourceCache<T, PreloadDataT>::HandleType ResourceCache<T, PreloadDataT>::getHandle(const std::string& path)
{
    return getResourceHandle(path);
}

template <typename T, typename PreloadDataT>
std::tuple<const T*, const PreloadDataT*, int, ResourceState> ResourceCache<T, PreloadDataT>::getResourceInfo(ResourceCache::HandleType h)
{
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);
    return std::tuple<const T*, const PreloadDataT*, int, ResourceState>(m_resources[h].resource.get(),m_resources[h].preloadData.get(),m_resources[h].resource.use_count()-1,m_resources[h].state);
}

template <typename T, typename PreloadDataT>
void ResourceCache<T, PreloadDataT>::doForEachResource(std::function<void(const std::string&, HandleType)> f)
{
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_rhmtx);
    for(const auto& mResource : m_resourceHandles)
    {
        f(mResource.first,mResource.second);
    }
}

template <typename T, typename PreloadDataT>
bool ResourceCache<T, PreloadDataT>::isReady(const std::string& path)
{
    HandleType h = getResourceHandle(path);
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);
    return (m_resources[h].state == ResourceState::ready || m_resources[h].state == ResourceState::defaulted || m_resources[h].state == ResourceState::failed);
}

template <typename T, typename PreloadDataT>
bool ResourceCache<T, PreloadDataT>::isPreloaded(const std::string& path)
{
    HandleType h = getResourceHandle(path);
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_rmtx);
    return !(m_resources[h].state == ResourceState::none || m_resources[h].state == ResourceState::preloading);
}

template <typename T, typename PreloadDataT>
bool ResourceCache<T, PreloadDataT>::loadOne()
{
    std::shared_lock<std::shared_timed_mutex> sharedLckRH(m_rhmtx);
    for(const auto& handle : m_resourceHandles)
    {
        HandleType h = handle.second;
        std::shared_lock<std::shared_timed_mutex> sharedLckR(m_rmtx);
        if(m_resources[h].state == ResourceState::preloaded ||  m_resources[h].state == ResourceState::preloadFailed)
        {
            sharedLckR.unlock();
            sharedLckRH.unlock();
            load(handle.first);
            return true;
        }
    }
    return false;
}

}
#endif //MPUTILS_RESOURCECACHE_H
