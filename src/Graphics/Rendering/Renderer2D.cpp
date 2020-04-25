/*
 * mpUtils
 * Renderer2D.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Renderer2D class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Rendering2D/Renderer2D.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

namespace {
constexpr unsigned char white[] = {0xFF,0xFF,0xFF,0xFF};
}

Renderer2D::Renderer2D(const std::string& shaderPath)
{
    addShaderIncludePath(shaderPath+"include/");
    m_spriteShader = ShaderProgram({{shaderPath+"sprite.vert"},
                                    {shaderPath+"sprite.frag"}});

    m_rectTexture = makeTexture2D(1,1,GL_RGBA8,white);

    setProjection(glm::mat4(1.0f));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    vao.bind();
}

void Renderer2D::setProjection(float left, float right, float bottom, float top, int minLayer, int maxLayer)
{
    setProjection(glm::ortho( left, right, bottom, top, float(minLayer), float(maxLayer)));
}

void Renderer2D::setProjection(const glm::mat4& projection)
{
    m_spriteShader.uniformMat4("projection", projection);
}

void Renderer2D::addRect(const glm::vec4& color, const glm::vec2& size, const glm::mat4& transform, int layer)
{
    glm::mat4 model = transform * glm::scale(glm::vec3(size/2,1.0f));
    model[3][2] = -layer;

    m_sprites.emplace_back(model,m_rectTexture.get(),color);
}

void Renderer2D::addSprite(const Sprite2D* sprite, const glm::mat4& transform, int layer, const glm::vec4& color)
{
    glm::mat4 model = transform * sprite->getBaseTransform();
    model[3][2] = -layer;

    m_sprites.emplace_back(model,&sprite->getTexture(),color);
}

void Renderer2D::render()
{
    m_spriteShader.use();
    for(const auto& sprite : m_sprites)
    {
        m_spriteShader.uniform4f("spriteColor",std::get<2>(sprite));
        m_spriteShader.uniformMat4("model",std::get<0>(sprite));
        std::get<1>(sprite)->bind(0);

        // draw (positions are calculated in the shader)
        glDrawArrays(GL_TRIANGLE_STRIP,0,4);
    }
    m_sprites.clear();
}


}}