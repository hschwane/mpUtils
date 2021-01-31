/*
 * mpUtils
 * Sprite2DTools.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/ContentCreation/Sprite2DTools.h"
#include "mpUtils/Misc/Image.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

Sprite2DData makeSimpleSprite(std::string pathToImage, std::string workDir)
{
    // load image
    auto img = Image8(workDir+pathToImage);

    // check for transparency
    bool hasSemiTransparancy=false;
    for(int i=0; i < img.numPixels(); ++i) {
        if( img(i)[3] > 0 && img(i)[3] < 255)
            hasSemiTransparancy = true;
    }

    // fill data
    Sprite2DData data;
    data.tileFactor = 1.0f;
    data.forward = 0.0f;
    data.texture = pathToImage;
    data.pivot = {0,0};
    data.rectInImage = {0,0,img.width(),img.height()};
    data.worldSize = {1.0f, float(img.height()) / float(img.width())};
    data.semiTransparent = hasSemiTransparancy;
    data.spritesheet = "";
    return data;
}

}}