/*
 * mpUtils
 * mpGraphicsResources.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MPGRAPHICSRESOURCES_H
#define MPUTILS_MPGRAPHICSRESOURCES_H

// includes
//--------------------
#include "mpUtils/Graphics/Rendering2D/Sprite2D.h"
#include "mpUtils/ResourceManager/Resource.h"
#include "mpUtils/ResourceManager/ResourceCache.h"
#include "mpUtils/ResourceManager/mpUtilsResources.h"
#include <memory>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// define resource bindings for Sprite2d and make sure templates can be precompiled
// usage:
// create the Resource managers type using the *RC types as resource caches
// the pass initializer list to the constructor and fill with the functions defined below
// example:
// ResourceManager< ImageRC > resourceManager( {preloadImage,finalLoadImage, /*default path*/,
//                                                      getDefaultImage(), /*name to show in ui*/} );

//!< data stored inbetween preload
struct Sprite2DPreloadData
{
    Image8 subImage;
    bool semiTransparent{false};
    glm::vec2 worldSize{1.0f, 1.0f};
    glm::vec2 pivot{0.0f, 0.0f};
    float forward{0.0f};
    float tileFactor{1.0f};
};

using Sprite2DResource = Resource<Sprite2D>; //!< resource to use sprite2D with the resource manager
using Sprite2DRC = ResourceCache<Sprite2D, Sprite2DPreloadData>; //!< resource cache to use sprite2D with the resource maanger

/**
 * @brief function to preload a 2d sprite in the resource manager,
 *          use std::bind to bind a pointer to the image cache component of your resource manager
 */
std::unique_ptr<Sprite2DPreloadData> preloadSprite(ImageRC*& imgrc, const std::string& data);
std::unique_ptr<Sprite2D> finalLoadSprite2D(std::unique_ptr<Sprite2DPreloadData> pd); //! function to create an actual sprite from the preloading data
std::unique_ptr<Sprite2D> getDefaultSprite(); //!< loads a default sprite, shown when sprites are missing

}
// instantiate some templates, so they can be linked
//-------------------------------------------------------------------
extern template class Resource<gph::Sprite2D>;
extern template class ResourceCache<gph::Sprite2D,gph::Sprite2DPreloadData>;

}
#endif //MPUTILS_MPGRAPHICSRESOURCES_H
