/*
 * mpUtils
 * imageLoading.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_IMAGELOADING_H
#define MPUTILS_IMAGELOADING_H

// includes
//--------------------
#include <memory>
#include <string>
#include "mpUtils/external/stb_image.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

/**
 * @brief load an image from file, enforcing four components per pixel (RGBA)
 * @param path the image file to load
 * @param width the with of the image is stored here
 * @param height the height of the image is stored here
 * @return a unique pointer to the image data
 */
std::unique_ptr<unsigned char[], void(*)(void*)> loadImageFile(const std::string& path, int& width, int& height, int forceChannels = STBI_rgb_alpha, bool flip = true);

/**
 * @brief load an hdr image from file, enforcing four components per pixel (RGBA)
 * @param path the image file to load
 * @param width the with of the image is stored here
 * @param height the height of the image is stored here
 * @return a unique pointer to the image data
 */
std::unique_ptr<float[], void(*)(void*)> loadImageFileHDR(const std::string& path, int& width, int& height, int forceChannels = STBI_rgb_alpha, bool flip=true);

/**
 * @brief load an image from from memory, enforcing four components per pixel (RGBA), can eg used together with the Resource.h.
 * @param data pointer to the data in memory
 * @param length number of bytes of data
 * @param path the image file to load
 * @param width the with of the image is stored here
 * @param height the height of the image is stored here
 * @return a unique pointer to the image data
 */
std::unique_ptr<unsigned char[], void(*)(void*)> loadImageData(const unsigned char * data, int length, int& width, int& height, int forceChannels = STBI_rgb_alpha, bool flip=true);

/**
 * @brief load an hdr image from memory, enforcing four components per pixel (RGBA), can eg used together with the Resource.h.
 * @param data pointer to the data in memory
 * @param length number of bytes of data
 * @param path the image file to load
 * @param width the with of the image is stored here
 * @param height the height of the image is stored here
 * @return a unique pointer to the image data
 */
std::unique_ptr<float[], void(*)(void*)> loadImageDataHDR(const unsigned char * data, int length, int& width, int& height, int forceChannels = STBI_rgb_alpha, bool flip=true);


}
#endif //MPUTILS_IMAGELOADING_H
