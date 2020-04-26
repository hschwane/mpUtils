/*
 * mpUtils
 * Sampler.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Sampler class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_SAMPLER_H
#define MPUTILS_SAMPLER_H

// includes
//--------------------
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <cinttypes>
#include <utility>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Sampler
 *
 * usage:
 * With a sampler, you can set several texture-independant parameters.
 * Functions are provided for commonly used parameters such as filter type and wrapping.
 *
 * You can also set other settings using the generic set() functions, eg:
 * GL_TEXTURE_MIN_LOD, GL_TEXTURE_MAX_LOD, GL_TEXTURE_LOD_BIAS, GL_TEXTURE_CUBE_MAP_SEAMLESS,
 * GL_TEXTURE_COMPARE_MODE, or GL_TEXTURE_COMPARE_FUNC
 *
 * You can use one sampler to control settings for several different textures.
 * use bind, to bind the sampler to a texture unit
 */
class Sampler
{
public:
    Sampler();
    ~Sampler();

    operator uint32_t() const; //!< convert to opengl handle for use in raw opengl functions

    // make sampler non copyable but movable
    Sampler(const Sampler& other) = delete;
    Sampler& operator=(const Sampler& other) = delete;
    Sampler(Sampler&& other) noexcept : m_handle(0){*this = std::move(other);};
    Sampler& operator=(Sampler&& other) noexcept {using std::swap; swap(m_handle,other.m_handle); return *this;};

    /**
     * @brief Set the filters used when downsampling or upsampling textures. Options:
     * GL_NEAREST
     * GL_LINEAR
     * GL_NEAREST_MIPMAP_NEAREST
     * GL_LINEAR_MIPMAP_NEAREST
     * GL_NEAREST_MIPMAP_LINEAR
     * GL_LINEAR_MIPMAP_LINEAR
     */
    void setFilter(GLint minFilter, GLint magFilter);

    /**
     * @brief set edge wrapping settings for all three directions of a texture Options:
     * GL_REPEAT
     * GL_MIRRORED_REPEAT
     * GL_CLAMP_TO_EDGE
     * GL_CLAMP_TO_BORDER
     * GL_MIRROR_CLAMP_TO_EDGE
     */
    void setWrap(GLint wrapS, GLint wrapT, GLint wrapR);

    void serBorderColor(const glm::vec4& color); //!< set a border color to be used with GL_CLAMP_TO_BORDER

    void set(GLenum parameter, GLint value) const; //!< set a parameter of type glint
    void set(GLenum parameter, const float* values) const; //!< set a paramter which requires multiple floats
    void bind(GLuint unit) const; //!< bind sampler to a texture unit

private:
    uint32_t m_handle;
};

}}

#endif //MPUTILS_SAMPLER_H
