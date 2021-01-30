/*
 * mpUtils
 * Tilemap.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Tilemap class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_TILEMAP_H
#define MPUTILS_TILEMAP_H

// includes
//--------------------
#include <random>
#include <glm/glm.hpp>
#include "mpUtils/Graphics/Rendering2D/TileType.h"
#include "mpUtils/Graphics/Rendering2D/Renderer2D.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Tilemap
 *
 * usage:
 * Create a tilemap, add tiles and call draw to render the tilemap.
 * Or use TilemapData and the resource manager to load a tilemap from a file
 *
 */
class Tilemap
{
public:
    Tilemap(glm::vec2 tileZeroPos, glm::vec2 spacing, glm::ivec2 size, AutotileMode autotile,
            const std::shared_ptr<TileType>& fillTile);

    void setTile(const glm::ivec2& pos, std::shared_ptr<TileType> type, int variant=-1); //!< set the tile type at pos
    const TileType& getTile(const glm::ivec2& pos) const; //!< get the tile type at pos
    glm::ivec2 getSize() const {return m_size;} //!< size of the tilemap

    /**
     * @brief add the tilemap to the render queue
     * @param renderer a 2d renderer to draw the map
     * @param layer the layer on which the map will be drawn
     *          transitions will be drawn at layer - precedence of the tile
     */
    void draw(Renderer2D& renderer, float layer);

private:
    AutotileMode m_autotileMode; //!< the mode of autotiling
    glm::vec2 m_spacing; //!< spacing between tile centers
    glm::ivec2 m_size; //! number of tiles in the map
    glm::vec2 m_tileZeroPos; //!< position of tile zero other tiles will be positive in x/y from that

    std::vector<std::shared_ptr<TileType>> m_tiles; //!< pointer to the tile for every place of the map
    std::vector<int> m_tileVariant; //!< stores a tile variant for every place in the map

    size_t tileId(const glm::ivec2& pos) const; //!< computes 1d tile id from 2d position

    std::default_random_engine rng;
    std::uniform_real_distribution<float> dist;
};

}}

#endif //MPUTILS_TILEMAP_H
