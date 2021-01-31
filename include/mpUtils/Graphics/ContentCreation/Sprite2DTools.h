/*
 * mpUtils
 * Sprite2DTools.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_SPRITE2DTOOLS_H
#define MPUTILS_SPRITE2DTOOLS_H

// includes
//--------------------
#include "mpUtils/Graphics/Rendering2D/Sprite2D.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

/**
 * @brief creates sprite data structure from image file using most obvious settings
 * @param pathToImage path to the image, this will be written to the sprite data
 * @param workDir added to the search path when opening the image file, but not written to the datastructure
 * @return the Sprite2DData datastructure for this sprite
 */
Sprite2DData makeSimpleSprite(std::string pathToImage, std::string workDir = "");

}}
#endif //MPUTILS_SPRITE2DTOOLS_H
