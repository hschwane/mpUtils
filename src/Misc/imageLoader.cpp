/*
 * mpUtils
 * imageLoader.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Misc/imageLoading.h"
#include "mpUtils/Log/Log.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

std::unique_ptr<unsigned char[], void(*)(void*)> loadImageFile(const std::string& path, int& width, int& height, int forceChannels, bool flip)
{
    stbi_set_flip_vertically_on_load(flip);

    int channels;
    auto result = std::unique_ptr<unsigned char[], decltype(&stbi_image_free)>(stbi_load(path.c_str(), &width, &height, &channels, forceChannels), &stbi_image_free);
    if(!result)
    {
        logERROR("ImageLoader") << "Error while loading image file "<< path << " reason: " << stbi_failure_reason();
        logFlush();
        throw std::runtime_error("Error while loading image file " + path + " reason: " + stbi_failure_reason());
    }
    return result;
}

std::unique_ptr<float[], void(*)(void*)> loadImageFileHDR(const std::string& path, int& width, int& height, int forceChannels, bool flip)
{
    stbi_set_flip_vertically_on_load(flip);

    int channels;
    auto result = std::unique_ptr<float[], decltype(&stbi_image_free)>(stbi_loadf(path.c_str(), &width, &height, &channels, forceChannels), &stbi_image_free);
    if(!result)
    {
        logERROR("ImageLoader") << "Error while loading image file "<< path << " reason: " << stbi_failure_reason();
        logFlush();
        throw std::runtime_error("Error while loading image file " + path + " reason: " + stbi_failure_reason());
    }
    return result;
}

std::unique_ptr<unsigned char[], void(*)(void*)> loadImageData(const unsigned char * data, int length, int& width, int& height, int forceChannels, bool flip)
{
    stbi_set_flip_vertically_on_load(flip);

    int channels;
    auto result = std::unique_ptr<unsigned char[], decltype(&stbi_image_free)>(stbi_load_from_memory( data, length, &width, &height, &channels, forceChannels), &stbi_image_free);
    if(!result)
    {
        logERROR("ImageLoader") << "Error while loading image data. Reason: " << stbi_failure_reason();
        logFlush();
        throw std::runtime_error(std::string("Error while loading image date. Reason: ") + stbi_failure_reason());
    }
    return result;
}

std::unique_ptr<float[], void(*)(void*)> loadImageDataHDR(const unsigned char * data, int length, int& width, int& height, int forceChannels, bool flip)
{
    stbi_set_flip_vertically_on_load(flip);

    int channels;
    auto result = std::unique_ptr<float[], decltype(&stbi_image_free)>(stbi_loadf_from_memory(data,length, &width, &height, &channels, forceChannels), &stbi_image_free);
    if(!result)
    {
        logERROR("ImageLoader") << "Error while loading image data. Reason: " << stbi_failure_reason();
        logFlush();
        throw std::runtime_error(std::string("Error while loading image date. Reason: ") + stbi_failure_reason());
    }
    return result;
}

}