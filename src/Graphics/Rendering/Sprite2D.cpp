/*
 * mpUtils
 * Sprite2D.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Sprite2D class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Rendering/Sprite2D.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Sprite2D class
//-------------------------------------------------------------------
Sprite2D::Sprite2D(const std::string& imageFile, glm::vec2 size, float forward)
{
    m_texture = makeTextureFromFile(imageFile);
    m_baseTransform = glm::scale(glm::mat4(1), glm::vec3(size/2,0.0f));
    m_baseTransform = glm::rotate(m_baseTransform, glm::two_pi<float>() - forward,glm::vec3{0.0f,0.0f,1.0f});
}

Sprite2D::Sprite2D(const unsigned char* data, int length, glm::vec2 size, float forward)
{
    m_texture = makeTextureFromData(data,length);
    m_baseTransform = glm::scale(glm::mat4(1), glm::vec3(size/2,0.0f));
    m_baseTransform = glm::rotate(m_baseTransform, glm::two_pi<float>() - forward,glm::vec3{0.0f,0.0f,1.0f});
}

}}
