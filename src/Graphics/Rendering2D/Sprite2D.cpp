/*
 * mpUtils
 * Sprite2D.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Sprite2D class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Rendering2D/Sprite2D.h"
#include "mpUtils/Log/Log.h"
#include "mpUtils/paths.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions
//-------------------------------------------------------------------

Sprite2D::Sprite2D(const std::string& imageFile, bool semiTransparent, const glm::vec2& worldSize, const glm::vec2& pivot,
                   float forward, float tileFactor)
{
    m_tileFactor = tileFactor;
    m_hasSemiTransparency = semiTransparent;

    try
    {
        m_texture = makeTextureFromFile(imageFile);
    } catch(const std::exception& e)
    {
        logERROR("Sprite2D") << "Could not load image " << imageFile << ". Error: " << e.what();
        m_texture = makeTextureFromFile(MPU_LIB_RESOURCE_PATH"missingTexture.png");
    }

    computeBaseTransform(worldSize,pivot,forward);
}

Sprite2D::Sprite2D(const Image8& image, bool semiTransparent, const glm::vec2& worldSize, const glm::vec2& pivot,
                   float forward, float tileFactor)
{
    m_tileFactor = tileFactor;
    m_hasSemiTransparency = semiTransparent;
    computeBaseTransform(worldSize,pivot,forward);
    m_texture = makeTextureFromImage(image);
}

void Sprite2D::computeBaseTransform(const glm::vec2& worldSize, const glm::vec2& pivot, float forward)
{
    m_baseTransform = glm::mat4(1.0);
    m_baseTransform = glm::rotate(m_baseTransform, glm::two_pi<float>() - forward,glm::vec3{0.0f,0.0f,1.0f});
    m_baseTransform = glm::scale(m_baseTransform, glm::vec3(worldSize*0.5f, 1.0f));
    m_baseTransform = glm::translate(m_baseTransform, glm::vec3(-pivot,0));
}

std::shared_ptr<Sprite2D> getEmptySprite()
{
    static auto _none_sprite = std::make_shared<mpu::gph::Sprite2D>(MPU_LIB_RESOURCE_PATH"none.png");
    return _none_sprite;
}

Sprite2DData::Sprite2DData(const std::string& tomlString)
{
    std::istringstream stream(tomlString);
    auto parsedData = toml::parse(stream);
    auto& sprite = toml::find(parsedData,"Sprite");

    displayName = toml::find<std::string>(sprite, "displayName");
    spritesheet = toml::find<std::string>(sprite, "spritesheet");
    texture = toml::find<std::string>(sprite, "texture");

    rectInImage.x = toml::find<int>(sprite, "rectInImage", 0);
    rectInImage.y = toml::find<int>(sprite, "rectInImage", 1);
    rectInImage.z = toml::find<int>(sprite, "rectInImage", 2);
    rectInImage.w = toml::find<int>(sprite, "rectInImage", 3);

    semiTransparent = toml::find<bool>(sprite, "semitransparent");

    worldSize.x = toml::find<float>(sprite, "worldSize", 0);
    worldSize.y = toml::find<float>(sprite, "worldSize", 1);

    pivot.x = toml::find<float>(sprite,"pivot", 0);
    pivot.y = toml::find<float>(sprite,"pivot", 1);

    forward = toml::find<float>(sprite, "forward");
    tileFactor = toml::find<float>(parsedData["Sprite"], "tileFactor");
}

toml::value Sprite2DData::toToml()
{
    toml::value tf(tileFactor);
    toml::value fw(forward);
    toml::value pv({pivot.x,pivot.y});
    toml::value ws({worldSize.x,worldSize.y});
    toml::value st(semiTransparent);
    toml::value ri({rectInImage.x,rectInImage.y,rectInImage.z,rectInImage.w});
    toml::value tex(texture);
    toml::value sprs(spritesheet);
    toml::value dspn(displayName);

    toml::value table({ {"tileFactor",tf}, {"forward",fw}, {"pivot",pv}, {"worldSize",ws}, {"semitransparent",st},
                        {"rectInImage",ri}, {"texture",tex}, {"spritesheet",sprs}, {"displayName",dspn} });
    return toml::value({ {"Sprite",table} });
}


}}
