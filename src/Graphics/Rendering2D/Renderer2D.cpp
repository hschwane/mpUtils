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
    m_spriteShader.uniformMat4("viewProjMat", m_projection * m_view);

    setSamplingLinear(true,true);

    // prepare colored rectangles
    m_rectTexture = makeTexture2D(1,1,GL_RGBA8,white);
    m_rectTextureHandle = m_rectTexture->getTexturehandleUvec2(m_sampler);
    m_rectTexture->makeTextureResident();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    vao.bind();
}

void Renderer2D::setProjection(float left, float right, float bottom, float top)
{
    m_projection = glm::ortho( left, right, bottom, top, 0.0f, 500.0f);
    m_viewProjection = m_projection * m_view;
    m_spriteShader.uniformMat4("viewProjMat", m_viewProjection);
}

void Renderer2D::setSamplingLinear(bool min, bool mag)
{
    GLint minFilter = GL_NEAREST_MIPMAP_NEAREST;
    GLint maxFilter = GL_NEAREST;

    if(min)
        minFilter=GL_LINEAR_MIPMAP_LINEAR;
    if(mag)
        maxFilter=GL_LINEAR;

    Sampler newSampler;
    newSampler.setWrap(GL_REPEAT,GL_REPEAT,GL_CLAMP_TO_EDGE);
    newSampler.setFilter(minFilter,maxFilter);
    m_sampler = std::move(newSampler);
}

void Renderer2D::setView(glm::mat4 view)
{
    m_view = view;
    m_viewProjection = m_projection * m_view;
    m_spriteShader.uniformMat4("viewProjMat", m_viewProjection);
}

const glm::mat4& Renderer2D::getViewProjection()
{
    return m_viewProjection;
}

void Renderer2D::addRect(const glm::vec4& color, const glm::vec2& size, const glm::mat4& transform, float layer)
{
    if(color.a < 0.00001f)
        return;

    glm::mat4 model = transform * glm::scale(glm::vec3(size/2,1.0f));
    model[3][2] = -layer;

    if(color.w < 9.9999f)
        m_transparentSprites.emplace_back(model,color,m_rectTextureHandle,1);
    else
        m_opaqueSprites.emplace_back(model,color,m_rectTextureHandle,1);
}

void Renderer2D::addSprite(const Sprite2D& sprite, const glm::mat4& transform, int layer, const glm::vec4& color)
{
    if(color.a < 0.00001f)
        return;

    glm::mat4 model = transform * sprite.getBaseTransform();
    model[3][2] = -layer;

    if( sprite.hasSemiTransparency() || color.a < 0.9999f )
        m_transparentSprites.emplace_back(model, color, sprite.getTexture().getTexturehandleUvec2(m_sampler),
                                          sprite.getTileFactor());

    else
        m_opaqueSprites.emplace_back(model,color, sprite.getTexture().getTexturehandleUvec2(m_sampler),sprite.getTileFactor());
    sprite.getTexture().makeTextureResident();
}

void Renderer2D::render()
{
    m_spriteShader.use();

    if(!m_opaqueSprites.empty())
    {
        // sort opaque sprites back to front, upload and draw
        std::sort(m_opaqueSprites.begin(), m_opaqueSprites.end(), [](const spriteData& a, const spriteData& b)
        {
            return a.model[3][2] > b.model[3][2];
        });

        m_spriteShader.uniform1b("alphaDiscard",true);
        Buffer<spriteData> opaque_data(m_opaqueSprites);
        opaque_data.bindBase(0, GL_SHADER_STORAGE_BUFFER);
        glDrawArrays(GL_TRIANGLES, 0, 6 * m_opaqueSprites.size());
        m_opaqueSprites.clear();
    }

    if(!m_opaqueSprites.empty())
    {
        // sort transparent sprites front to back, upload and draw
        std::sort(m_transparentSprites.begin(), m_transparentSprites.end(), [](const spriteData& a, const spriteData& b)
        {
            return a.model[3][2] < b.model[3][2];
        });

        glDepthMask(GL_FALSE);
        m_spriteShader.uniform1b("alphaDiscard", false);
        Buffer<spriteData> transparent_data(m_transparentSprites);
        transparent_data.bindBase(0, GL_SHADER_STORAGE_BUFFER);
        glDrawArrays(GL_TRIANGLES, 0, 6 * m_transparentSprites.size());
        m_transparentSprites.clear();
        glDepthMask(GL_TRUE);
    }
}

}}