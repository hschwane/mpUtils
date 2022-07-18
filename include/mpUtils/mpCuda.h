/*
 * mpUtils
 * mpCuda.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MPCUDA_H
#define MPUTILS_MPCUDA_H

// only include this file in *.cu files
//--------------------
#ifndef __CUDACC__
    #error "Only use the cudaUtils.h if compiling *.cu files with nvcc!"
#endif
//--------------------

// includes
//--------------------

// cuda stuff from the framework
#include "Cuda/cudaUtils.h"
#include "mpUtils/external/cuda/helper_math.h"
#include "Cuda/helper_math_missing.h"
#include "Cuda/Matrix.h"
#include "Cuda/MatrixMath.h"
#include "Cuda/MemoryModifiers.h"
#include "Cuda/VectorReference.h"
#include "Cuda/ManagedAllocator.h"
#include "Cuda/PinnedAllocator.h"
#include "Cuda/DeviceVector.h"
#include "Cuda/GlBufferMapper.h"
//--------------------

#endif //MPUTILS_MPCUDA_H
