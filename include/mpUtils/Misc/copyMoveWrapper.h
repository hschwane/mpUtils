/*
 * mpUtils
 * AtomicWrapper.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the AtomicWrapper class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_COPYMOVEWRAPPER_H
#define MPUTILS_COPYMOVEWRAPPER_H

// includes
//--------------------
#include <atomic>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

/**
 * @brief makes a movable functor copyable using a shared pointer
 * @tparam F type of functor
 * @param f functor to make copyable
 * @return the copyable object
 */
template<typename F >
auto makeFuncCopyable( F&& f );

//-------------------------------------------------------------------
/**
 * class CopyMoveAtomic
 *
 * A copy and movable wrapper for an atomic. The copy and move operations semselfs are not atomic!
 * But this way atomics can be stored in a container.
 *
 */
template <typename T>
class CopyMoveAtomic : public std::atomic<T>
{
public:
    CopyMoveAtomic()
            : std::atomic<T>()
    {}

    CopyMoveAtomic(T t)
            : std::atomic<T>(t)
    {}

    CopyMoveAtomic &operator=(T t)
    {
        std::atomic<T>::store(t);
        return *this;
    }

    CopyMoveAtomic(const std::atomic<T>& a)
            : std::atomic<T>(a.load())
    {}

    CopyMoveAtomic(const CopyMoveAtomic& other)
            : std::atomic<T>(other.load())
    {}

    CopyMoveAtomic(CopyMoveAtomic&& other)
            : std::atomic<T>(std::move(other.load()))
    {}

    CopyMoveAtomic &operator=(const CopyMoveAtomic& other)
    {
        std::atomic<T>::store(other.load());
    }

    CopyMoveAtomic &operator=(CopyMoveAtomic&& other)
    {
        std::atomic<T>::store(std::move(other.load()));
    }
};

//-------------------------------------------------------------------
// function definitions
template<typename F >
auto makeFuncCopyable( F&& f )
{
    auto spf = std::make_shared<F>(std::forward<F>(f) );
    return [spf](auto&&... args)->decltype(auto)
    {
        return (*spf)( (args)... );
    };
}


}
#endif //MPUTILS_COPYMOVEWRAPPER_H
