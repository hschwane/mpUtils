/*
 * mpUtils
 * Tilemap.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Tilemap class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Rendering2D/Tilemap.h"
#include "mpUtils/Misc/pointPicking.h"
#include "mpUtils/Log/Log.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Tilemap class
//-------------------------------------------------------------------
Tilemap::Tilemap(glm::vec2 tileZeroPos, glm::vec2 spacing, glm::ivec2 size, AutotileMode autotile,
                 const std::shared_ptr<TileType>& fillTile)
        : m_autotileMode(autotile), m_spacing(spacing), m_size(size), m_tileZeroPos(tileZeroPos),
          m_tiles(size.x*size.y,fillTile), m_tileVariant(size.x*size.y,0),
          rng(getRanndomSeed()), dist(0,1)
{
}

void Tilemap::setTile(const glm::ivec2& pos, std::shared_ptr<TileType> type, int variant)
{
    if(variant < 0)
        variant = type->selectVariant(dist(rng));
    size_t tile = tileId(pos);
    m_tiles[tile] = type;
    m_tileVariant[tile] = variant;
}

const TileType& Tilemap::getTile(const glm::ivec2& pos) const
{
    return *m_tiles[tileId(pos)];
}

void Tilemap::draw(Renderer2D& renderer, float layer)
{
    switch(m_autotileMode) {
        case AutotileMode::corners:
        {

            break;
        }
        case AutotileMode::blobInner:
        {

            break;
        }
        case AutotileMode::blobOuter:
        {

            break;
        }
        case AutotileMode::none:
        {
            glm::ivec2 t;
            for(t.y = 0; t.y < m_size.y; ++t.y)
                for(t.x = 0; t.x < m_size.x; ++t.x) {
                    size_t id = tileId(t);
                    glm::mat4 tf = glm::translate(glm::vec3({m_tileZeroPos+(glm::vec2(t)*m_spacing), 0}));
                    // draw base sprite
                    int v = m_tileVariant[id];
                    renderer.addSprite(m_tiles[id]->getSprite(v,0),tf,layer);
            }
            break;
        }
        default:
            logERROR("Tilemap") << "Autotile mode " << static_cast<int>(m_autotileMode) << "not supported!";
    }
}

size_t Tilemap::tileId(const glm::ivec2& pos) const
{
    return pos.y * m_size.x + pos.x;
}


}}