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
#include "mpUtils/Graphics/Rendering2D/Sprite2D.h"
#include "mpUtils/Log/Log.h"
#include "mpUtils/paths.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Sprite2D class
//-------------------------------------------------------------------
Sprite2D::Sprite2D(const std::string& imageFile, bool semiTransparent, glm::vec2 size, float forward, float tileFactor)
{
    m_tileFactor = tileFactor;
    m_hasSemiTransparency = semiTransparent;

    try
    {
        m_texture = makeTextureFromFile(imageFile);
    } catch(const std::exception& e)
    {
        logERROR("Sprite2D") << "Could not load image " << imageFile << ". Error: " << e.what();
        m_texture = makeTextureFromFile(MPU_LIB_RESOURCE_PATH"missingTexture.png");
    }

    m_baseTransform = glm::scale(glm::mat4(1), glm::vec3(size/2,1.0f));
    m_baseTransform = glm::rotate(m_baseTransform, glm::two_pi<float>() - forward,glm::vec3{0.0f,0.0f,1.0f});
}

//Sprite2D::Sprite2D(const unsigned char* data, int length, bool semiTransparent, glm::vec2 size, float forward, float tileFactor)
//{
//    m_tileFactor = tileFactor;
//    m_hasSemiTransparency = semiTransparent;
//
//    try
//    {
//        m_texture = makeTextureFromData(data,length);
//    } catch(const std::exception& e)
//    {
//        logERROR("Sprite2D") << "Could not load image from data. Error: " << e.what();
//        m_texture = makeTextureFromFile(MPU_LIB_RESOURCE_PATH"missingTexture.png");
//    }
//
//    m_baseTransform = glm::scale(glm::mat4(1), glm::vec3(size/2,0.0f));
//    m_baseTransform = glm::rotate(m_baseTransform, glm::two_pi<float>() - forward,glm::vec3{0.0f,0.0f,1.0f});
//}

}}
