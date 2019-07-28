/*
 * mpUtils
 * Sampler.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Sampler class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Opengl/Sampler.h"
#include "glm/ext.hpp"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Sampler class
//-------------------------------------------------------------------
Sampler::Sampler()
{
    glCreateSamplers(1,&m_handle);
}

Sampler::~Sampler()
{
    glDeleteSamplers(1,&m_handle);
}

Sampler::operator uint32_t() const
{
    return m_handle;
}

void Sampler::setFilter(GLint minFilter,GLint magFilter)
{
    glSamplerParameteri(*this, GL_TEXTURE_MAG_FILTER, magFilter);
    glSamplerParameteri(*this, GL_TEXTURE_MIN_FILTER, minFilter);
}

void Sampler::setWrap(GLint wrapS,GLint wrapT,GLint wrapR)
{
    glSamplerParameteri(*this, GL_TEXTURE_WRAP_S, wrapS);
    glSamplerParameteri(*this, GL_TEXTURE_WRAP_T, wrapT);
    glSamplerParameteri(*this, GL_TEXTURE_WRAP_R, wrapR);
}

void Sampler::serBorderColor(const glm::vec4 &color)
{
    glSamplerParameterfv(*this, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(color));
}

void Sampler::set(const GLenum parameter, const int value) const
{
    glSamplerParameteri(*this, parameter, value);
}

void Sampler::set(const GLenum parameter, const float* values) const
{
    glSamplerParameterfv(*this, parameter, values);
}

void Sampler::bind(GLuint unit) const
{
    glBindSampler(unit,*this);
}

}}