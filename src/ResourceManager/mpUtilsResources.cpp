/*
 * mpUtils
 * mpUtilsResources.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/ResourceManager/mpUtilsResources.h"
#include "mpUtils/paths.h"
#include <memory>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

// function definitions
//-------------------------------------------------------------------
std::unique_ptr<Image8> preloadImage(std::string data)
{
    return std::make_unique<Image8>(reinterpret_cast<const unsigned char*>(data.data()), data.size());
}

std::unique_ptr<Image8> finalLoadImage(std::unique_ptr<Image8> img)
{
    return img;
}

std::unique_ptr<Image8> getDefaultImage()
{
    return std::make_unique<Image8>(MPU_LIB_RESOURCE_PATH "missingTexture.png");
}

std::unique_ptr<Image16> preloadImage16(std::string data)
{
    return std::make_unique<Image16>(reinterpret_cast<const unsigned char*>(data.data()), data.size());
}

std::unique_ptr<Image16> finalLoadImage16(std::unique_ptr<Image16> img)
{
    return img;
}

std::unique_ptr<Image16> getDefaultImage16()
{
    return std::make_unique<Image16>(MPU_LIB_RESOURCE_PATH "missingTexture.png");
}

std::unique_ptr<Image32> preloadImage32(std::string data)
{
    return std::make_unique<Image32>(reinterpret_cast<const unsigned char*>(data.data()), data.size());
}

std::unique_ptr<Image32> finalLoadImage32(std::unique_ptr<Image32> img)
{
    return img;
}

std::unique_ptr<Image32> getDefaultImage32()
{
    return std::make_unique<Image32>(MPU_LIB_RESOURCE_PATH "missingTexture.png");
}

// instantiate some templates, so they can be linked
//-------------------------------------------------------------------
template class Resource<Image8>;
template class ResourceCache<Image8, Image8>;
template class Resource<Image16>;
template class ResourceCache<Image16, Image16>;
template class Resource<Image32>;
template class ResourceCache<Image32, Image32>;

}