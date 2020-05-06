/*
 * mpUtils
 * Image.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Misc/Image.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

// instantiate templates so they can be linked
//-------------------------------------------------------------------
template class Image<uint8_t>;
template class Image<uint16_t>;
template class Image<float>;

}