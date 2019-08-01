/*
 * mpUtils
 * Renderer2D.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Renderer2D class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_RENDERER2D_H
#define MPUTILS_RENDERER2D_H

// includes
//--------------------
#include <string>
#include <glm/glm.hpp>
#include "mpUtils/Graphics/Utils/Transform2D.h"
#include "mpUtils/Graphics/Rendering/SpriteInstance2D.h"
#include "mpUtils/paths.h"
#include "mpUtils/Graphics/Opengl/Shader.h"
#include "mpUtils/Graphics/Opengl/VertexArray.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Renderer2D
 *
 * usage:
 * Class takes care of rendering all sorts of 2d graphics.
 *
 */
class Renderer2D
{
public:
    Renderer2D(const std::string& shaderPath = MPU_LIB_SHADER_PATH);

    void setProjection(const glm::mat4& projection);
    void setProjection(float left, float right, float bottom, float top, int minLayer, int maxLayer);
    void addSprite(const SpriteInstance2D& sprite, const glm::mat4& transform=glm::mat4(1.0f), int layer=0);
    void render();

private:
    ShaderProgram m_spriteShader;
    std::vector< std::pair<glm::mat4,const SpriteInstance2D*>> m_sprites;
    VertexArray vao;
};

}}

#endif //MPUTILS_RENDERER2D_H
