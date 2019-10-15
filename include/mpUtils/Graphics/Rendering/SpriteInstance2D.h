/*
 * mpUtils
 * SpriteInstance2D.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the SpriteInstance2D class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_SPRITEINSTANCE2D_H
#define MPUTILS_SPRITEINSTANCE2D_H

// includes
//--------------------
#include "mpUtils/Graphics/Rendering/Sprite2D.h"
#include <memory>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class SpriteInstance2D
 *
 * usage:
 * Sprite instance is used to create multiple instances of the same sprite which can have different settings for rendering.
 *
 */
class SpriteInstance2D
{
public:

    /**
     * @brief create a sprite instance by passing a shared pointer to a sprite and an optional color
     * @param sprite the sprite that is internally used when rendering
     * @param color color to be multiplied with the texture value
     */
    SpriteInstance2D(std::shared_ptr<Sprite2D> sprite, glm::vec4 color=glm::vec4(1.0f));

    /**
     * @brief set the color of the sprite
     * @param color color is multiplied with the original texture color
     */
    void setColor(glm::vec4 color) { m_color=color;}

    /**
     * @brief references the sprite object used by this sprite instance
     */
    Sprite2D& getSprite() const { return *m_sprite;}

    /**
     * @return the color with which the sprite is colorized
     */
    glm::vec4 getColor() const { return m_color;}

private:
    std::shared_ptr<Sprite2D> m_sprite;
    glm::vec4 m_color;
};

}}

#endif //MPUTILS_SPRITEINSTANCE2D_H
