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
#include <memory>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include "mpUtils/Graphics/Opengl/Texture.h"
#include "mpUtils/Misc/Image.h"
#include "mpUtils/external/toml/toml.hpp"
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
 * Class represents a sprite object. The same sprite can be drawn at different position using the renderer 2D.
 *
 */
class Sprite2D
{
public:
    /**
     * @brief Create Sprite by loading image from file
     * @param bool set to true if alpha is not 0 or 255, so semi transparent areas are rendered correctly
     * @param imageFile path to the image file
     * @param worldSize size of the sprite in world coordinates
     * @param pivot point where further translations are applied
     * @param forward forward direction in radians, 0 means to the right pi to the left
     * @param tileFactor texture will be tiled that many times
     */
    explicit Sprite2D(const std::string& imageFile, bool semiTransparent=false, const glm::vec2& worldSize={1, 1},
            const glm::vec2& pivot={0,0}, float forward=0, float tileFactor=1);

    /**
     * @brief Create Sprite from the given image
     * @param bool set to true if alpha is not 0 or 255, so semi transparent areas are rendered correctly
     * @param image path to the image file
     * @param worldSize size of the sprite in world coordinates
     * @param pivot point where further translations are applied
     * @param forward forward direction in radians, 0 means to the right pi to the left
     * @param tileFactor texture will be tiled that many times
     */
    explicit Sprite2D(const Image8& image, bool semiTransparent=false, const glm::vec2& worldSize={1, 1},
                      const glm::vec2& pivot={0,0}, float forward=0, float tileFactor=1);

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

    /**
     * @brief returns if the texture hat alpha values different from 0/255
     */
    bool hasSemiTransparency() const {return  m_hasSemiTransparency;}

private:
    void computeBaseTransform(const glm::vec2& worldSize, const glm::vec2& pivot,float forward);

    std::unique_ptr<Texture> m_texture; //!< texture to be used
    glm::mat4 m_baseTransform; //!< transformation matrix to transform a 1x1 quad into correct size and direction
    float m_tileFactor; //!< texture will be tiled that many times
    bool m_hasSemiTransparency; //!< does the texture have semi transparent values?
};

//-------------------------------------------------------------------
// helper functions
std::shared_ptr<Sprite2D> getEmptySprite(); //!< returns an empty sprite

//-------------------------------------------------------------------
// helper to load and store sprites
/**
 * class Sprite2DData
 * @brief stores all data known stored in a .sprite file and can read from a string / store into a string
 *          errors during parsing will throw an exception
 */
class Sprite2DData
{
public:
    Sprite2DData() = default; //!< set some maybe useable defaults
    explicit Sprite2DData(const std::string& tomlString); //!< parse toml string into the struct
    toml::value toToml(); //!< store into a tomle struct, that can be serialized to a string or file

    std::string displayName;
    std::string spritesheet;
    std::string texture;
    glm::ivec4 rectInImage{0,0,1,1};
    bool semiTransparent{false};
    glm::vec2 worldSize{1.0f,1.0f};
    glm::vec2 pivot{0.0f,0.0f};
    float forward{0.0f};
    float tileFactor{1.0f};
};

}}

#endif //MPUTILS_SPRITE2D_H
