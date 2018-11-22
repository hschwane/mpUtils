/*
 * mpUtils
 * matrix3x3.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MATRIX3X3_H
#define MPUTILS_MATRIX3X3_H

// includes
//--------------------
#include <iostream>
#include <type_traits>
#include <sstream>
#include "mpUtils/Misc/type_traitUtils.h"
#include "mpUtils/version.h"
#ifdef MPU_GLM_AVAILIBLE
    #include <glm/glm.hpp>
#endif
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
 * Class Template Mat
 * This is a class template for small matrices to be used with cuda on host or device.
 * On host, glm matrix will be faster due to vectorization.
 * @tparam TYPE the internal data type needs to be an arithmetic type
 * @tparam numRows number of rows
 * @tparam numCols number of columns
 */
template<typename TYPE, size_t numRows, size_t numCols>
class Mat
{
    static_assert(std::is_arithmetic<TYPE>::value, "Non arithmetic type used for matrix.");
public:
    // default constructors
    Mat() = default;

    // additional construction
    CUDAHOSTDEV explicit Mat(const TYPE v); //!< constructor fills the diagonal with v

    template <typename... cArgs, std::enable_if_t< (sizeof...(cArgs) > 1) && (sizeof...(cArgs) == numRows*numCols), int> = 0>
    CUDAHOSTDEV explicit Mat(const cArgs... v) : m_data{static_cast<TYPE>(v)...} {} //!< constructs matrix with a value for each element


#ifdef MPU_GLM_AVAILIBLE
    // conversion to glm
    template<glm::qualifier Q>
    explicit Mat(glm::mat<numRows, numCols, TYPE, Q> &glmat); //!< constructs this from glm matrix
    template<glm::qualifier Q>
    explicit operator glm::mat<numRows, numCols, TYPE, Q>(); //!< convert to glm matrix
#endif

    // data access
    CUDAHOSTDEV TYPE *operator[](size_t row) { return &m_data[numCols * row]; } //!< access a row
    CUDAHOSTDEV const TYPE *operator[](size_t row) const { return &m_data[numCols * row]; } //!< access a row

    CUDAHOSTDEV TYPE &operator()(size_t idx) { return m_data[idx]; } //!< access value
    CUDAHOSTDEV const TYPE &operator()(size_t idx) const { return m_data[idx]; } //!< access value

    CUDAHOSTDEV TYPE &T(size_t row, size_t col) { return this[col][row];} //!< access value as if matrix was transposed
    CUDAHOSTDEV const TYPE &T(size_t row, size_t col) const { return this[col][row];} //!< access value as if matrix was transposed

    // logical operators
    CUDAHOSTDEV bool operator==(const Mat &other) const;
    CUDAHOSTDEV bool operator!=(const Mat &other) const;

    // arithmetic operators
    CUDAHOSTDEV Mat &operator+=(const Mat &other); //!< component wise addition
    CUDAHOSTDEV Mat &operator-=(const Mat &other); //!< component wise subtraction
    CUDAHOSTDEV Mat operator+(const Mat &other) const; //!< component wise addition
    CUDAHOSTDEV Mat operator-(const Mat &other) const; //!< component wise subtraction

    CUDAHOSTDEV Mat &operator*=(const TYPE &v); //!< scalar multiply
    CUDAHOSTDEV Mat &operator/=(const TYPE &v); //!< scalar divide
    CUDAHOSTDEV Mat operator*(const TYPE &v) const; //!< scalar multiply
    CUDAHOSTDEV Mat operator/(const TYPE &v) const; //!< scalar divide

    CUDAHOSTDEV Mat &operator*=(const Mat &rhs); //!< matrix multiplication
    template<size_t rhsRows, size_t rhsCols>
    CUDAHOSTDEV Mat<TYPE, numRows, rhsCols> operator*(const Mat<TYPE, rhsRows, rhsCols> &rhs) const; //!< matrix multiplication

    static constexpr size_t size = numRows * numCols;
    static constexpr size_t cols = numCols;
    static constexpr size_t rows = numRows;
private:
    TYPE m_data[size];
};

/**
 * @brief scalar multiplication is order independent
 */
template<typename T, size_t rows, size_t cols>
CUDAHOSTDEV Mat<T, rows, cols> operator*(const T &lhs, const Mat<T, rows, cols> &rhs);


/**
 * @brief convert a matrix to string for debugging
 */
template<typename T, size_t rows, size_t cols>
std::string toString(const Mat<T,rows,cols>& mat);

// define all the template functions of the matrix class
//-------------------------------------------------------------------

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols>::Mat(const TYPE v)
{
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
        {
            if(i == j)
                (*this)[i][j] = v;
            else
                (*this)[i][j] = 0;
        }
}

#ifdef MPU_GLM_AVAILIBLE
template<typename TYPE, size_t numRows, size_t numCols>
template<glm::qualifier Q>
Mat<TYPE, numRows, numCols>::Mat(glm::mat<numRows, numCols, TYPE, Q> &glmat)
{
    for(int i = 0; i < numRows; i++)
        for(int j = 0; j < numCols; j++)
        {
            (*this)[i][j] = glmat[i][j];
        }
}

template<typename TYPE, size_t numRows, size_t numCols>
template<glm::qualifier Q>
Mat<TYPE, numRows, numCols>::operator glm::mat<numRows, numCols, TYPE, Q>()
{
    glm::mat<numRows, numCols, TYPE, Q> r;

    for(int i = 0; i < numRows; i++)
        for(int j = 0; j < numCols; j++)
        {
            r[i][j] = (*this)[i][j];
        }

    return r;
}

#endif

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV bool Mat<TYPE, rows, cols>::operator==(const Mat &other) const
{
    for(int i = 0; i < size; ++i)
    {
        if(m_data[i] != other.m_data[i])
            return false;
    }
    return true;
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV bool Mat<TYPE, rows, cols>::operator!=(const Mat &other) const
{
    return !((*this) == other);
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> &Mat<TYPE, rows, cols>::operator+=(const Mat &other)
{
    for(int i = 0; i < size; ++i)
    {
        m_data[i] += other.m_data[i];
    }
    return *this;
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> &Mat<TYPE, rows, cols>::operator-=(const Mat &other)
{
    for(int i = 0; i < size; ++i)
    {
        m_data[i] -= other.m_data[i];
    }
    return *this;
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> Mat<TYPE, rows, cols>::operator+(const Mat &other) const
{
    Mat<TYPE, rows, cols> temp(*this);
    temp += other;
    return temp;
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> Mat<TYPE, rows, cols>::operator-(const Mat &other) const
{
    Mat<TYPE, rows, cols> temp(*this);
    temp -= other;
    return temp;
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> &Mat<TYPE, rows, cols>::operator*=(const TYPE &v)
{
    for(int i = 0; i < size; ++i)
    {
        m_data[i] *= v;
    }
    return *this;
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> &Mat<TYPE, rows, cols>::operator/=(const TYPE &v)
{
    for(int i = 0; i < size; ++i)
    {
        m_data[i] /= v;
    }
    return *this;
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> Mat<TYPE, rows, cols>::operator*(const TYPE &v) const
{
    Mat<TYPE, rows, cols> temp(*this);
    temp *= v;
    return temp;
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> Mat<TYPE, rows, cols>::operator/(const TYPE &v) const
{
    Mat<TYPE, rows, cols> temp(*this);
    temp /= v;
    return temp;
}

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> &Mat<TYPE, rows, cols>::operator*=(const Mat &rhs)
{
    Mat tmp(*this);

    for(int i = 0; i < rows; ++i)
        for(int k = 0; k < cols; ++k)
        {
            (*this)[i][k] = tmp[i][0] * rhs[0][k];
            for(int j = 1; j < cols; ++j)
            {
                (*this)[i][k] += tmp[i][j] * rhs[j][k];
            }
        }

    return *this;
}

template<typename TYPE, size_t rows, size_t cols>
template<size_t rhsRows, size_t rhsCols>
CUDAHOSTDEV Mat<TYPE, rows, rhsCols> Mat<TYPE, rows, cols>::operator*(const Mat<TYPE, rhsRows, rhsCols> &rhs) const
{
    static_assert(cols == rhsRows, "Matrices of these sizes can not be multiplied.");
    Mat<TYPE, rows, rhsCols> result(0);

    for(int i = 0; i < rows; ++i)
        for(int k = 0; k < rhsCols; ++k)
        {
            result[i][k] = (*this)[i][0] * rhs[0][k];
            for(int j = 1; j < cols; ++j)
            {
                result[i][k] += (*this)[i][j] * rhs[j][k];
            }
        }

    return result;
}

// define the helper functions
//-------------------------------------------------------------------

template<typename TYPE, size_t rows, size_t cols>
CUDAHOSTDEV Mat<TYPE, rows, cols> operator*(const TYPE &lhs, const Mat<TYPE, rows, cols>& rhs)
{
    return rhs*lhs;
}

template<typename TYPE, size_t rows, size_t cols>
std::string toString(const Mat<TYPE,rows,cols>& mat)
{
    std::ostringstream ss;
    for(int i = 0; i < rows; ++i)
    {
        ss << "| " << mat[i][0];
        for(int j = 1; j < cols; ++j)
        {
            ss << ",  " << mat[i][j];
        }
        ss << " |\n";
    }
    return ss.str();
}

}

#endif //MPUTILS_MATRIX3X3_H
