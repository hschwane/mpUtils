/*
 * mpUtils
 * Sprite2DTools.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_SPRITE2DTOOLS_H
#define MPUTILS_SPRITE2DTOOLS_H

// includes
//--------------------
#include "mpUtils/Graphics/Rendering2D/Sprite2D.h"
#include "mpUtils/Misc/pointPicking.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

/**
 * @brief creates sprite data structure from image file using most obvious settings
 * @param pathToImage path to the image, this will be written to the sprite data
 * @param workDir added to the search path when opening the image file, but not written to the datastructure
 * @return the Sprite2DData datastructure for this sprite
 */
Sprite2DData makeSimpleSprite(std::string pathToImage, std::string workDir = "");

/**
 * @brief controls an imgui window that allows sprite and spritesheet editing
 */
class SpriteEditor
{
public:
    void show(bool* show = nullptr, bool drawAsChild = false);
//    void setSprite(std::string filename, Sprite2DData data);
private:
    bool m_hasUnsavedChanges{false};
    void tryLoadTexture();
    void selectTextureWithFileDlg();
    void autoDetectTransparancy();
    void setCropToFullImage();
    void autoFillAll();
    void drawImageOverlay(const glm::vec2& previewStartPos, const glm::vec2& previewSize);

    // settings
    std::string m_workDir;
    const std::string m_id{"###"+std::to_string(getRanndomSeed())};

    // current sprite data
    std::string m_filename; //!< current sprites filename
    Sprite2DData m_data; //!< current sprites data
    std::unique_ptr<Image8> m_image; //!< the image the texture was loaded from
    std::unique_ptr<Texture> m_texture; //!< current sprites texture;

};

}}
#endif //MPUTILS_SPRITE2DTOOLS_H
