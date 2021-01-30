/*
 * mpUtils
 * TileType.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the TileType class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_TILETYPE_H
#define MPUTILS_TILETYPE_H

// includes
//--------------------
#include <unordered_map>
#include "mpUtils/Graphics/Rendering2D/Sprite2D.h"
#include "mpUtils/Misc/unordered_map2d.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

enum class AutotileMode {
    none=0,
    corners=1,
    blob=2,
};

//-------------------------------------------------------------------
/**
 * class TileType
 *
 * a type of tile used on the tile map
 * can have different variations for the zero tile, transitions tiles always have the same sprite
 *
 */
class TileType
{
public:
    //!< create a new tile type using textures
    TileType(std::string refName, std::vector<float> frequencies, float precedence,
             unordered_map2d<int,int,std::shared_ptr<Sprite2D>> sprites);

    //!< create a simple tile without variants or transitions
    TileType(std::string refName, std::shared_ptr<mpu::gph::Sprite2D> displaySprite)
            : m_displayName(std::move(refName)), m_precedence(0)
    {
        assert_critical(displaySprite, "TileType", "Constructing TileType from nullptr sprite!");
        m_sprites[std::make_pair(0,0)] = displaySprite;
        m_frequencies.push_back(1.0);
    }

    const std::string& getName() const { return m_displayName; } //!< the name of this tile type in the user interface
    int selectVariant(float r) const; //!< pass random number to select a variant according to frequency

    /**
     * @brief [0,1) tiles with higher precedence are displayed on top in transitions
     */
    float getPrecedence() const { return m_precedence; }

    /**
     * @brief returns sprite for the desired variant and transition bitmask
     * @param var sprite variant
     * @param bitmask transition bitmask
     * @return a reference to the correct sprite
     */
    const Sprite2D& getSprite(int var, int bitmask) const;

private:
    unordered_map2d<int,int,std::shared_ptr<Sprite2D>> m_sprites; //!< all sprites sorted by variant and bitmask
    std::string m_displayName; //!< the name of the tile as displayed in the ui
    std::vector<float> m_frequencies; //!< frequencies of the different tile variants
    float m_precedence; //!< tiles with higher precedence are displayed on top in transitions
};

//-------------------------------------------------------------------
/**
 * class TileData
 * @brief stores all data from a .tile file and can read from a string / store into a string
 *          errors during parsing will throw an exception
 */
class TileData
{
public:
    TileData() = default;
    explicit TileData(const std::string& toml);
    toml::value toToml();

    std::string displayName;
    AutotileMode autotileMode;
    std::vector<std::pair<std::pair<int,int>,std::string>> spriteFilenames;
    std::vector<float> frequencies;
    float precedence;
};

}}

#endif //MPUTILS_TILETYPE_H
