/*
 * mpUtils
 * unordered_map2d.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_UNORDERED_MAP2D_H
#define MPUTILS_UNORDERED_MAP2D_H

// includes
//--------------------
#include <unordered_map>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

namespace detail {
    struct hash_pair
    {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2>& p) const
        {
            auto hash1 = std::hash<T1>{}(p.first);
            auto hash2 = std::hash<T2>{}(p.second);
            return hash1 ^ hash2;
        }
    };
}

template<typename Key1, typename Key2, typename Val>
using unordered_map2d = std::unordered_map<std::pair<Key1,Key2>,Val, detail::hash_pair>;

}}

#endif //MPUTILS_UNORDERED_MAP2D_H
