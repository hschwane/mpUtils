/*
 * mpUtils
 * MatrixMath.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 * defines template functions to provide additional operations to the matrix class
 *
 */
#ifndef MPUTILS_MATRIXMATH_H
#define MPUTILS_MATRIXMATH_H

// includes
//--------------------
#include "Matrix.h"
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

// some additional math operations
//--------------------

/**
 * @brief calculate the transpose of matrix m
 *          actually transposes in memory by hard copy
 */
template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, cols, rows> transpose(Mat<T, rows, cols> &m);

/**
 * @brief performes component wise multiplication of two matrices of same size
 */
template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> compWiseMult(Mat<T, rows, cols> &first, Mat<T, rows, cols> &second);

// determinant of matrix up to 4D
//--------------------

/**
 * @brief calculates the determinant of m
 */
template<typename T>
CUDAHOSTDEV T determinant(const Mat<T, 2, 2> &m);

/**
 * @brief calculates the determinant of m
 */
template<typename T>
CUDAHOSTDEV T determinant(const Mat<T, 3, 3> &m);

/**
 * @brief calculates the determinant of m
 */
template<typename T>
CUDAHOSTDEV T determinant(const Mat<T, 4, 4> &m);

// invert matrix up to 4D
//--------------------

/**
 * @brief calculates the inverse matrix undefined if determinant is zero
 */
template<typename T>
CUDAHOSTDEV Mat<T, 2, 2> invert(const Mat<T, 2, 2> &m);

 /**
  * @brief calculates the inverse matrix undefined if determinant is zero
  */
template <typename T>
 CUDAHOSTDEV Mat<T,3,3> invert(const Mat<T,3,3> &m);

/**
 * @brief calculates the inverse matrix undefined if determinant is zero
 */
template<typename T>
CUDAHOSTDEV Mat<T, 4, 4> invert(const Mat<T, 4, 4> &m);


// matrix vector product for up to 4D
//--------------------

// helper to check if TYPE has a x attribute
namespace detail {
template<class T>
using hasx_t = decltype(std::declval<T>().x);
}

/**
 * @brief multiply a 2D vector with a 2x2 matrix
 */
template<typename T, typename vT, std::enable_if_t<
        !std::is_same<T, vT>::value && mpu::is_detected<detail::hasx_t, vT>(), int> = 0>
CUDAHOSTDEV vT operator*(Mat<T, 2, 2> lhs, vT &rhs);

/**
 * @brief multiply a 3D vector with a 3x3 matrix
 */
template<typename T, typename vT, std::enable_if_t<
        !std::is_same<T, vT>::value && mpu::is_detected<detail::hasx_t, vT>(), int> = 0>
CUDAHOSTDEV vT operator*(Mat<T, 3, 3> lhs, vT &rhs);

/**
 * @brief multiply a 4D vector with a 4x4 matrix
 */
template<typename T, typename vT, std::enable_if_t<
        !std::is_same<T, vT>::value && mpu::is_detected<detail::hasx_t, vT>(), int> = 0>
CUDAHOSTDEV vT operator*(Mat<T, 4, 4> lhs, vT &rhs);


// define all the helper functions
//-------------------------------------------------------------------

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, cols, rows> transpose(Mat<T, rows, cols> &m)
{
    Mat<T, cols, rows> result;
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            result[j][i] = m[i][j];
    return result;
}

template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> compWiseMult(Mat<T, rows, cols> &first, Mat<T, rows, cols> &second)
{
    Mat<T,rows,cols> r;

    for(int i = 0; i < r.size; ++i)
        r(i) = first(i)*second(i);

    return r;
}

template<typename T>
CUDAHOSTDEV T determinant(const Mat<T, 2, 2> &m)
{
    return m(0)*m(3) - m(1)*m(2);
}

template<typename T>
CUDAHOSTDEV T determinant(const Mat<T, 3, 3> &m)
{
    return m(0)*m(4)*m(8) + m(1)*m(5)*m(6) + m(2)*m(3)*m(7) -
           m(6)*m(4)*m(2) - m(7)*m(5)*m(0) - m(8)*m(1)*m(3);
}

template<typename T>
CUDAHOSTDEV T determinant(const Mat<T, 4, 4> &m)
{
    return  m(12) * m(9)  * m(6)  * m(3)   -  m(8) * m(13) * m(6)  * m(3)   -
            m(12) * m(5)  * m(10) * m(3)   +  m(4) * m(13) * m(10) * m(3)   +
            m(8)  * m(5)  * m(14) * m(3)   -  m(4) * m(9)  * m(14) * m(3)   -
            m(12) * m(9)  * m(2)  * m(7)   +  m(8) * m(13) * m(2)  * m(7)   +
            m(12) * m(1)  * m(10) * m(7)   -  m(0) * m(13) * m(10) * m(7)   -
            m(8)  * m(1)  * m(14) * m(7)   +  m(0) * m(9)  * m(14) * m(7)   +
            m(12) * m(5)  * m(2)  * m(11)  -  m(4) * m(13) * m(2)  * m(11)  -
            m(12) * m(1)  * m(6)  * m(11)  +  m(0) * m(13) * m(6)  * m(11)  +
            m(4)  * m(1)  * m(14) * m(11)  -  m(0) * m(5)  * m(14) * m(11)  -
            m(8)  * m(5)  * m(2)  * m(15)  +  m(4) * m(9)  * m(2)  * m(15)  +
            m(8)  * m(1)  * m(6)  * m(15)  -  m(0) * m(9)  * m(6)  * m(15)  -
            m(4)  * m(1)  * m(10) * m(15)  +  m(0) * m(5)  * m(10) * m(15);
}

template<typename T>
CUDAHOSTDEV Mat<T, 2, 2> invert(const Mat<T, 2, 2> &m)
{
    Mat<T,2,2> r;
    T det = determinant(m);
    det = 1.0/det;

    r(0) = m(3) *det;
    r(1) = -m(1) *det;
    r(2) = -m(2) *det;
    r(3) = m(0) *det;

    return r;
}

template<typename T>
CUDAHOSTDEV Mat<T, 3, 3> invert(const Mat<T, 3, 3> &m)
{
    Mat<T, 3, 3> r;
    T det = determinant(m);
    det = 1 / det;

    r(0) = (m(4) * m(8) - m(7) * m(5)) * det;
    r(1) = (m(2) * m(7) - m(1) * m(8)) * det;
    r(2) = (m(1) * m(5) - m(2) * m(4)) * det;
    r(3) = (m(5) * m(6) - m(3) * m(8)) * det;
    r(4) = (m(0) * m(8) - m(2) * m(6)) * det;
    r(5) = (m(3) * m(2) - m(0) * m(5)) * det;
    r(6) = (m(3) * m(7) - m(6) * m(4)) * det;
    r(7) = (m(6) * m(1) - m(0) * m(7)) * det;
    r(8) = (m(0) * m(4) - m(3) * m(1)) * det;

    return r;
}

template<typename T>
CUDAHOSTDEV Mat<T, 4, 4> invert(const Mat<T, 4, 4> &m)
{
    Mat<T, 4, 4> inv;

    inv(0) = m(5)  * m(10) * m(15) -
             m(5)  * m(11) * m(14) -
             m(9)  * m(6)  * m(15) +
             m(9)  * m(7)  * m(14) +
             m(13) * m(6)  * m(11) -
             m(13) * m(7)  * m(10);

    inv(4) = -m(4)  * m(10) * m(15) +
             m(4)  * m(11) * m(14) +
             m(8)  * m(6)  * m(15) -
             m(8)  * m(7)  * m(14) -
             m(12) * m(6)  * m(11) +
             m(12) * m(7)  * m(10);

    inv(8) = m(4)  * m(9) * m(15) -
             m(4)  * m(11) * m(13) -
             m(8)  * m(5) * m(15) +
             m(8)  * m(7) * m(13) +
             m(12) * m(5) * m(11) -
             m(12) * m(7) * m(9);

    inv(12) = -m(4)  * m(9) * m(14) +
              m(4)  * m(10) * m(13) +
              m(8)  * m(5) * m(14) -
              m(8)  * m(6) * m(13) -
              m(12) * m(5) * m(10) +
              m(12) * m(6) * m(9);

    inv(1) = -m(1)  * m(10) * m(15) +
             m(1)  * m(11) * m(14) +
             m(9)  * m(2) * m(15) -
             m(9)  * m(3) * m(14) -
             m(13) * m(2) * m(11) +
             m(13) * m(3) * m(10);

    inv(5) = m(0)  * m(10) * m(15) -
             m(0)  * m(11) * m(14) -
             m(8)  * m(2) * m(15) +
             m(8)  * m(3) * m(14) +
             m(12) * m(2) * m(11) -
             m(12) * m(3) * m(10);

    inv(9) = -m(0)  * m(9) * m(15) +
             m(0)  * m(11) * m(13) +
             m(8)  * m(1) * m(15) -
             m(8)  * m(3) * m(13) -
             m(12) * m(1) * m(11) +
             m(12) * m(3) * m(9);

    inv(13) = m(0)  * m(9) * m(14) -
              m(0)  * m(10) * m(13) -
              m(8)  * m(1) * m(14) +
              m(8)  * m(2) * m(13) +
              m(12) * m(1) * m(10) -
              m(12) * m(2) * m(9);

    inv(2) = m(1)  * m(6) * m(15) -
             m(1)  * m(7) * m(14) -
             m(5)  * m(2) * m(15) +
             m(5)  * m(3) * m(14) +
             m(13) * m(2) * m(7) -
             m(13) * m(3) * m(6);

    inv(6) = -m(0)  * m(6) * m(15) +
             m(0)  * m(7) * m(14) +
             m(4)  * m(2) * m(15) -
             m(4)  * m(3) * m(14) -
             m(12) * m(2) * m(7) +
             m(12) * m(3) * m(6);

    inv(10) = m(0)  * m(5) * m(15) -
              m(0)  * m(7) * m(13) -
              m(4)  * m(1) * m(15) +
              m(4)  * m(3) * m(13) +
              m(12) * m(1) * m(7) -
              m(12) * m(3) * m(5);

    inv(14) = -m(0)  * m(5) * m(14) +
              m(0)  * m(6) * m(13) +
              m(4)  * m(1) * m(14) -
              m(4)  * m(2) * m(13) -
              m(12) * m(1) * m(6) +
              m(12) * m(2) * m(5);

    inv(3) = -m(1) * m(6) * m(11) +
             m(1) * m(7) * m(10) +
             m(5) * m(2) * m(11) -
             m(5) * m(3) * m(10) -
             m(9) * m(2) * m(7) +
             m(9) * m(3) * m(6);

    inv(7) = m(0) * m(6) * m(11) -
             m(0) * m(7) * m(10) -
             m(4) * m(2) * m(11) +
             m(4) * m(3) * m(10) +
             m(8) * m(2) * m(7) -
             m(8) * m(3) * m(6);

    inv(11) = -m(0) * m(5) * m(11) +
              m(0) * m(7) * m(9) +
              m(4) * m(1) * m(11) -
              m(4) * m(3) * m(9) -
              m(8) * m(1) * m(7) +
              m(8) * m(3) * m(5);

    inv(15) = m(0) * m(5) * m(10) -
              m(0) * m(6) * m(9) -
              m(4) * m(1) * m(10) +
              m(4) * m(2) * m(9) +
              m(8) * m(1) * m(6) -
              m(8) * m(2) * m(5);

    T det = m(0) * inv(0) + m(1) * inv(4) + m(2) * inv(8) + m(3) * inv(12);
    det = 1.0 / det;
    inv *= det;

    return inv;
}

template<typename T, typename vT, std::enable_if_t<!std::is_same<T,vT>::value && mpu::is_detected<detail::hasx_t,vT>(), int>>
CUDAHOSTDEV vT operator*(Mat<T, 2, 2> lhs, vT &rhs)
{
    return vT{lhs(0) * rhs.x + lhs(1) * rhs.y,
              lhs(2) * rhs.x + lhs(3) * rhs.y};
}

template<typename T, typename vT, std::enable_if_t<!std::is_same<T,vT>::value && mpu::is_detected<detail::hasx_t,vT>(), int>>
CUDAHOSTDEV vT operator*(Mat<T, 3, 3> lhs, vT &rhs)
{
    return vT{lhs(0) * rhs.x + lhs(1) * rhs.y + lhs(2) * rhs.z,
              lhs(3) * rhs.x + lhs(4) * rhs.y + lhs(5) * rhs.z,
              lhs(6) * rhs.x + lhs(7) * rhs.y + lhs(8) * rhs.z};
}

template<typename T, typename vT, std::enable_if_t<!std::is_same<T,vT>::value && mpu::is_detected<detail::hasx_t,vT>(), int>>
CUDAHOSTDEV vT operator*(Mat<T, 4, 4> lhs, vT &rhs)
{
    return vT{lhs(0) * rhs.x + lhs(1) * rhs.y + lhs(2) * rhs.z + lhs(3) * rhs.w,
              lhs(4) * rhs.x + lhs(5) * rhs.y + lhs(6) * rhs.z + lhs(7) * rhs.w,
              lhs(8) * rhs.x + lhs(9) * rhs.y + lhs(10) * rhs.z + lhs(11) * rhs.w,
              lhs(12) * rhs.x + lhs(13) * rhs.y + lhs(14) * rhs.z + lhs(15) * rhs.w};
}



}
#endif //MPUTILS_MATRIXMATH_H
