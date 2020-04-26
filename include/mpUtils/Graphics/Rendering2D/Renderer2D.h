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
#include "Sprite2D.h"
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
    /**
     * @brief create a 2d renderer
     * @param shaderPath path where the shader files are located
     */
    explicit Renderer2D(const std::string& shaderPath = MPU_LIB_SHADER_PATH);

    /**
     * @brief Sets a (orthographic) projection matrix to be used when rendering.
     *          Layers outside your clipping planes values will not be shown.
     * @param projection the matrix to use for rendering
     */
    void setProjection(const glm::mat4& projection);

    /**
     * @brief enable / disable linear sampling for filtering (default enabled)
     */
    void setSamplingLinear(bool min, bool mag);

    /**
     * @brief Create am orthographic projection matrix to be used when rendering.
     * @param left where the left edge of the viewport should be in world coordinates
     * @param right where the right edge of the viewport should be in world coordinates
     * @param bottom where the left edge of the viewport should be in world coordinates
     * @param top where the top edge of the viewport should be in world coordinates
     * @param minLayer smallest layer id to be drawn
     * @param maxLayer biggest layer id to be drawn
     */
    void setProjection(float left, float right, float bottom, float top, int minLayer, int maxLayer);

    /**
     * @brief draws a filled rectangle
     * @param transform the transform that is applied to the sprite before rendering
     * @param layer the layer the sprite should be rendered in
     * @param color add tint color
     */
    void addRect(const glm::vec4& color=glm::vec4(1.0f), const glm::vec2& size=glm::vec2(1.0f), const glm::mat4& transform=glm::mat4(1.0f), int layer=0);

    /**
     * @brief add a sprite to be rendered this frame
     * @param sprite the sprite to be rendered, sprite must be valid until render() is called
     * @param transform the transform that is applied to the sprite before rendering
     * @param layer the layer the sprite should be rendered in
     * @param color add tint color
     */
    void addSprite(const Sprite2D* sprite, const glm::mat4& transform=glm::mat4(1.0f), int layer=0, const glm::vec4& color=glm::vec4(1.0f));

    /**
     * @brief render everything queued up for this frame then clear queue
     */
    void render();

private:

    //!< internally stores all data needed for rendering of a sprite
    struct spriteData
    {
        spriteData(const glm::mat4& m, const glm::vec4& c, const glm::uvec2& texAdr, int texIdx, float tf)
                : model(m), color(c), bindlessTexture(texAdr), textureArrayIndex(texIdx), tileFactor(tf) {}
        glm::mat4 model;
        glm::vec4 color;
        glm::uvec2 bindlessTexture;
        unsigned int textureArrayIndex;
        float tileFactor;
    };

    Sampler m_sampler; //!< the sampler used to sample textures
    std::unique_ptr<Texture> m_rectTexture; //!< white texture for colored rectangles
    glm::uvec2 m_rectTextureHandle; //!< bindless handle of the rect texture

    ShaderProgram m_spriteShader; //!< shader to be used for rendering quads
    std::vector<spriteData> m_sprites;
    VertexArray vao;
};

}}

#endif //MPUTILS_RENDERER2D_H
