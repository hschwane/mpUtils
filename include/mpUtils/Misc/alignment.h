/*
 * mpUtils
 * alignment.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_ALIGNMENT_H
#define MPUTILS_ALIGNMENT_H

#if defined(__CUDACC__) // NVCC
    #define MPU_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
    #define MPU_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MPU_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MPU_ALIGN macro for your host compiler!"
#endif

#endif //MPUTILS_ALIGNMENT_H
