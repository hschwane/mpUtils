/*
 * mpUtils
 * callbackUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_CALLBACKUTILS_H
#define MPUTILS_CALLBACKUTILS_H

// includes
//--------------------
#include <vector>
//--------------------

// includes
//--------------------
namespace mpu {
//--------------------

template <typename T, typename F>
int addCallback(std::vector <std::pair<int, T>>& callbackVector, F f)
{
    int id;
    if(callbackVector.empty())
        id = 0;
    else
        id = callbackVector.back().first + 1;

    callbackVector.emplace_back(id, f);
    return id;
}

template <typename T>
void removeCallback(std::vector <std::pair<int, T>>& callbackVector, int id)
{
    auto it = std::lower_bound(callbackVector.cbegin(), callbackVector.cend(), std::pair<int, T>(id, T{}),
                               [](const std::pair<int, T>& a, const std::pair<int, T>& b)
                               { return (a.first < b.first); });
    if(it != callbackVector.end())
        callbackVector.erase(it);
}

}
#endif //MPUTILS_CALLBACKUTILS_H
