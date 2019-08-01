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
#include "mpUtils/Graphics/Rendering/Renderer2D.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

Renderer2D::Renderer2D(const std::string& shaderPath)
{
    addShaderIncludePath(shaderPath+"include/");
    m_spriteShader = ShaderProgram({{shaderPath+"sprite.vert"},
                                    {shaderPath+"sprite.frag"}});
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

void Renderer2D::addSprite(const SpriteInstance2D& sprite, const glm::mat4& transform, int layer)
{
    glm::mat4 model = transform * sprite.getSprite().getBaseTransform();
    model[3][2] = -layer;

    m_sprites.emplace_back(model,&sprite);
}

void Renderer2D::render()
{
    m_spriteShader.use();
    for(const auto& sprite : m_sprites)
    {
        m_spriteShader.uniform4f("spriteColor",sprite.second->getColor());
        m_spriteShader.uniformMat4("model",sprite.first);
        sprite.second->getSprite().getTexture().bind(GL_TEXTURE0);

        // draw (positions are calculated in the shader)
        glDrawArrays(GL_TRIANGLE_STRIP,0,4);
    }
    m_sprites.clear();
}


}}