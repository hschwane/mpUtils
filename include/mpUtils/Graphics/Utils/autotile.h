/*
 * mpUtils
 * autotile.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_AUTOTILE_H
#define MPUTILS_AUTOTILE_H

// includes
//--------------------
#include <array>
#include <map>
#include <cinttypes>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

/**
 * @brief compute an autotile bitmap from a reference element and the 4 elements in the neighborhood
 * @param neighborhood the elements in the neighborhood N-W-E-S
 * @param reference the reference element, ie the element in the middle
 */
template<typename T>
unsigned int inline autotileBlob4(const std::array<T,4>& neighborhood, T reference)
{
    unsigned int bitmask = 0;
    for(int i = 0; i < 4; ++i)
        if(neighborhood[i] == reference)
            bitmask |= (1 << i);

    return bitmask;
}

/**
 * @brief compute an autotile bitmap from a reference element and the 8 elements in the neighborhood
 * @param neighborhood the elements in the neighborhood NW-N-NE-W-E-SW-S-SE
 * @param reference the reference element, ie the element in the middle
 */
template<typename T>
unsigned int inline autotileBlob8(const std::array<T,8>& neighborhood, T reference)
{
    unsigned int bitmask = 0;
    for(int i = 0; i < 4; ++i)
    {
        int e = i*2+1;
        int c = i*2;

        if(neighborhood[e] == reference)
            bitmask |= (1 << i);

        if(neighborhood[c] == reference && neighborhood[(c - 1) % 8] == reference && neighborhood[(c + 1) % 8] == reference )
            bitmask |= (1 << (i+4));
    }
    return bitmask;
}

/**
 * @brief compute an autotile bitmap, depending on which corner matches the reference
 * @param neighborhood the values at the four corners NW-NE-SW-SE
 * @param reference the reference element, ie what the corners need to match
 */
template<typename T>
unsigned int inline autotileCorner(const std::array<T,4>& neighborhood, T reference)
{
    unsigned int bitmask = 0;
    for(int i = 0; i < 4; ++i)
        if(neighborhood[i] == reference)
            bitmask |= (1 << i);

    return bitmask;
}

//template<typename T>
//std::map<T,unsigned int> inline multiAutotileBlob4(const std::array<T,4>& neighborhood)
//{
//    std::map<T,unsigned int> bitmasks;
//    uint8_t bitmask = 0;
//    for(int i = 0; i < 4; ++i)
//        if(neighborhood[i] == reference)
//            bitmask[] |= (1 << i);
//
//    return bitmask;
//}
//
//template<typename T>
//std::map<T,unsigned int> inline multiAutotileBlob8(const std::array<T,8>& neighborhood, T reference)
//{
//    std::map<T,unsigned int> bitmasks;
//    for(int i = 0; i < 4; ++i)
//    {
//        int e = i*2+1;
//        int c = i*2;
//
//        if(neighborhood[e] == reference)
//            bitmask |= (1 << i);
//
//        if(neighborhood[c] == reference & && neighborhood[(c - 1) % 8] == reference && neighborhood[(c + 1) % 8] == reference )
//            bitmask |= (1 << (i+4));
//    }
//    return bitmask;
//}
//
///**
// * @brief compute an autotile bitmap, depending on the four corners
// * @param value at the four corners NW-NE-SW-SE
// * @return map of all surrounding elements and the resulting bitmask
// */
//template<typename T>
//std::map<T,unsigned int> inline multiAutotileCorner(const std::array<T,4>& neighborhood)
//{
//    std::map<T,unsigned int> bitmasks;
//    for(int i = 0; i < 4; ++i)
//        bitmask[neighborhood[i]] |= (1 << i);
//    return bitmask;
//}

}}

#endif //MPUTILS_AUTOTILE_H
