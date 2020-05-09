/*
 * mpUtils
 * mpGraphicsResources.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include <mpUtils/ResourceManager/mpGraphicsResources.h>

#include "mpUtils/ResourceManager/mpUtilsResources.h"
#include "mpUtils/paths.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

std::unique_ptr<Sprite2DPreloadData>
preloadSprite(ImageRC*& imgrc, const std::string& data)
{
    Sprite2DData spd(data);
    auto image = imgrc->load(spd.texture);
    auto preloadData = std::make_unique<Sprite2DPreloadData>();

    preloadData->subImage = image->cloneSubregionGCoord(spd.rectInImage.x,spd.rectInImage.y,spd.rectInImage.z,spd.rectInImage.w);
    preloadData->pivot = spd.pivot;
    preloadData->semiTransparent = spd.semiTransparent;
    preloadData->worldSize = spd.worldSize;
    preloadData->tileFactor = spd.tileFactor;
    preloadData->forward = spd.forward;

    return preloadData;
}

std::unique_ptr<Sprite2D> finalLoadSprite2D(std::unique_ptr<Sprite2DPreloadData> pd)
{
    return std::make_unique<Sprite2D>(pd->subImage,pd->semiTransparent,pd->worldSize,pd->pivot,pd->forward,pd->tileFactor);
}

std::unique_ptr<Sprite2D> getDefaultSprite()
{
    return std::make_unique<Sprite2D>(MPU_LIB_RESOURCE_PATH "missingSprite.png");
}

}
// instantiate some templates, so they can be linked
//-------------------------------------------------------------------
template class Resource<gph::Sprite2D>;
template class ResourceCache<gph::Sprite2D,gph::Sprite2DPreloadData>;

}