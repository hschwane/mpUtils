/*
 * mpUtils
 * TileType.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the TileType class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Rendering2D/TileType.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the TileType class
//-------------------------------------------------------------------
TileType::TileType(std::string refName, std::vector<float> frequencies, float precedence,
                   unordered_map2d<int,int,std::shared_ptr<Sprite2D>>  sprites)
        : m_displayName(std::move(refName)),  m_frequencies(std::move(frequencies)), m_precedence(precedence),
            m_sprites(std::move(sprites))

{
    for(const auto& x : sprites)
        assert_critical(x.second, "TileType", "Constructing TileType" + refName + "from nullptr sprite!");

    if(frequencies.empty())
        frequencies.push_back(1.0f);
}

int TileType::selectVariant(float r) const
{
    float p = 0;
    for(int i = 0; i < m_frequencies.size(); ++i)
    {
        p += m_frequencies[i];
        if(r <= p)
            return i;
    }
    return 0;
}

const mpu::gph::Sprite2D& TileType::getSprite(int var, int bitmask) const
{
    auto s = m_sprites.find(std::make_pair(var,bitmask));

    while(s == m_sprites.end() && var > 0) {
        s = m_sprites.find(std::make_pair(--var,bitmask));
    }

    if(s == m_sprites.end())
        return *getEmptySprite();

    return *(s->second);
}

// function definitions of the TileData class
//-------------------------------------------------------------------
TileData::TileData(const std::string& toml)
{
    std::istringstream stream(toml);
    auto parsedData = toml::parse(stream);
    auto& tile = toml::find(parsedData, "Tile");

    displayName = toml::find<std::string>(tile, "displayName");
    autotileMode = static_cast<AutotileMode>(toml::find<int>(tile, "autotileMode"));

    auto spriteArray = toml::find<toml::array>(tile, "sprites");
    spriteFilenames.reserve(spriteArray.size());

    int maxv =0;
    for(const auto& item : spriteArray) {
        int v = toml::find<int>(item, "v");
        int b = toml::find<int>(item, "b");
        std::string s = toml::find<std::string>(item, "s");
        spriteFilenames.emplace_back(std::make_pair(std::make_pair(v,b),s));
        maxv = std::max(maxv,v);
    }

    ++maxv;
    frequencies = toml::find_or<std::vector<float>>(tile, "frequencies", std::vector<float>(maxv,1.0 / maxv));
    precedence = toml::find<float>(tile, "precedence");
}

toml::value TileData::toToml()
{
    toml::value dn(displayName);
    toml::array sp;
    toml::value fr(frequencies);
    toml::value pd(precedence);

    for(const auto& item : spriteFilenames) {
        toml::value v(item.first.first);
        toml::value b(item.first.second);
        toml::value s(item.second);
        toml::table tbl( {{"s",s}, {"b",b}, {"v",v}} );
        sp.emplace_back(tbl);
    }

    toml::value table({{"precedence",  pd},
                       {"frequencies", fr},
                       {"sprites",     sp},
                       {"displayName", dn}});
    return toml::value({{"Tile", table}});
}


}}