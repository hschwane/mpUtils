/*
 * mpUtils
 * additionalMath.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_ADDITIONALMATH_H
#define MPUTILS_ADDITIONALMATH_H

// includes
//--------------------
#include <cmath>
//--------------------

// this file contains device/host functions that also need to compile when using gcc
//--------------------
#ifndef CUDAHOSTDEV
#ifdef __CUDACC__
    #define CUDAHOSTDEV __host__ __device__
#else
    #define CUDAHOSTDEV
#endif
#endif
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

/**
 * @brief convert degrees to radians
 */
template <typename T>
constexpr T rad(const T degree) noexcept
{
    return degree * T(M_PI / 180);
}

/**
 * @brief convert radians to degree
 */
template <typename T>
constexpr T deg(const T radians) noexcept
{
    return radians * T(180 * M_1_PI);
}


}
#endif //MPUTILS_ADDITIONALMATH_H
