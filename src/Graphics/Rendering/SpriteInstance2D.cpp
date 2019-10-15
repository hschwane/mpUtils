/*
 * mpUtils
 * SpriteInstance2D.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the SpriteInstance2D class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Rendering/SpriteInstance2D.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the SpriteInstance2D class
//-------------------------------------------------------------------
SpriteInstance2D::SpriteInstance2D(std::shared_ptr<Sprite2D> sprite, glm::vec4 color)
        : m_sprite(sprite),
        m_color(color)
{
}

}}