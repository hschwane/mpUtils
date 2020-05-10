/*
 * mpUtils
 * mpUtilsResources.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MPUTILSRESOURCES_H
#define MPUTILS_MPUTILSRESOURCES_H

// includes
//--------------------
#include "mpUtils/Misc/Image.h"
#include "mpUtils/ResourceManager/Resource.h"
#include "mpUtils/ResourceManager/ResourceCache.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

// define resource bindings for image and make sure templates can be precompiled
// usage:
// create the Resource managers type using the *RC types as resource caches
// the pass initializer list to the constructor and fill with the functions defined below
// example:
// ResourceManager< ImageRC > resourceManager( {preloadImage,finalLoadImage, /*default path*/,
//                                                      getDefaultImage(), /*name to show in ui*/} );

using ImageResource = Resource<Image8>; //!< resource to use 8bit image with the resource manager
using ImageRC = ResourceCache<Image8,Image8>; //!< resource cache to use 8bit image with the resource manager
std::unique_ptr<Image8> preloadImage(std::string data); //!< function to preload an 8bit image in the resource manager
std::unique_ptr<Image8> finalLoadImage(std::unique_ptr<Image8> img); //!< finalize loading of a preloaded 8bit image in the resource manager
std::unique_ptr<Image8> getDefaultImage(); //!< loads a default image to be passed to the resource manager

using Image16Resource = Resource<Image16>; //!< resource to use 16bit image with the resource manager
using Image16RC = ResourceCache<Image16,Image16>; //!< resource cache to use 16bit image with the resource manager
std::unique_ptr<Image16> preloadImage16(std::string data); //!< function to preload an 16bit image in the resource manager
std::unique_ptr<Image16> finalLoadImage16(std::unique_ptr<Image16> img); //!< finalize loading of a preloaded 16bit image in the resource manager
std::unique_ptr<Image16> getDefaultImage16(); //!< loads a default image to be passed to the resource manager

using Image32Resource = Resource<Image32>; //!< resource to use 32bit image with the resource manager
using Image32RC = ResourceCache<Image32,Image32>; //!< resource cache to use 32bit image with the resource manager
std::unique_ptr<Image32> preloadImage32(std::string data); //!< function to preload an 32bit image in the resource manager
std::unique_ptr<Image32> finalLoadImage32(std::unique_ptr<Image32> img); //!< finalize loading of a preloaded 32bit image in the resource manager
std::unique_ptr<Image32> getDefaultImage32(); //!< loads a default image to be passed to the resource manager

// instantiate some templates, so they can be linked
//-------------------------------------------------------------------
//extern template class Resource<Image8>;
//extern template class ResourceCache<Image8,Image8>;
//extern template class Resource<Image16>;
//extern template class ResourceCache<Image16,Image16>;
//extern template class Resource<Image32>;
//extern template class ResourceCache<Image32,Image32>;

}
#endif //MPUTILS_MPUTILSRESOURCES_H
