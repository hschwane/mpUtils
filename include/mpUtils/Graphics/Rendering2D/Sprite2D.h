/*
 * mpUtils
 * Sprite2D.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Sprite2D class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_SPRITE2D_H
#define MPUTILS_SPRITE2D_H

// includes
//--------------------
#include "mpUtils/Graphics/Opengl/Texture.h"
#include <memory>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class sprite2D
 *
 * usage:
 * Class represents a sprite object. All data that is shared between multiple sprite instances is stored here.
 *
 */
class Sprite2D
{
public:
    /**
     * @brief Create Sprite loading image from file
     * @param imageFile path to the image file
     * @param size size of the sprite in world coordinates
     * @param forward forward direction in radians, 0 means to the right pi to the left
     * @param tileFactor texture will be tiled that many times
     */
    Sprite2D(const std::string& imageFile, glm::vec2 size={1,1}, float forward=0, float tileFactor=1);

    /**
     * @brief Create Sprite loading image from memory (eg resource file)
     * @param imageFile path to the image file
     * @param size size of the sprite in world coordinates
     * @param forward forward direction in radians, 0 means to the right pi to the left
     * @param tileFactor texture will be tiled that many times
     */
    Sprite2D(const unsigned char * data, int length, glm::vec2 size={1,1}, float forward=0, float tileFactor=1);

    /**
     * @brief returns a transformation matrix to transform a 1x1 quad into correct size and direction
     */
    glm::mat4 getBaseTransform() const { return m_baseTransform;}

    /**
     * @brief references the texture used by this sprite
     */
    Texture& getTexture() const {return *m_texture;}

    /**
     * @brief returns how many times the texture will be tiled inside the sprite
     */
    float getTileFactor() const {return m_tileFactor;}

private:
    std::unique_ptr<Texture> m_texture; //!< texture to be used
    glm::mat4 m_baseTransform; //!< transformation matrix to transform a 1x1 quad into correct size and direction
    float m_tileFactor; //!< texture will be tiled that many times
};

}}

#endif //MPUTILS_SPRITE2D_H
