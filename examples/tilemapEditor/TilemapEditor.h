/*
 * mpUtils
 * TilemapEditor.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the TilemapEditor class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_TILEMAPEDITOR_H
#define MPUTILS_TILEMAPEDITOR_H

// includes
//--------------------
#include <mpUtils/mpUtils.h>
#include <mpUtils/mpGraphics.h>

#if defined(__GNUC__) && __GNUC__ < 8
    #include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
    namespace fs = std::filesystem;
#endif
//--------------------

//!< sprite with metadata
struct metaSprite
{
    fs::path filename;
    mpu::gph::Sprite2DData data;
    std::shared_ptr<mpu::gph::Sprite2D> sprite;
};

//!< TileType with metadata
struct metaTile
{
    fs::path filename;
    mpu::gph::TileData data;
    mpu::gph::TileType tile;
};

// setup resource manager
using ResourceManagerType = mpu::ResourceManager<mpu::ImageRC,mpu::gph::Sprite2DRC>;
ResourceManagerType& getRM(); //!< returns the resource management system
extern template class mpu::ResourceManager<mpu::ImageRC,mpu::gph::Sprite2DRC>;

//-------------------------------------------------------------------
/**
 * class TilemapEditor
 *
 * usage:
 *
 */
class TilemapEditor
{
public:
    TilemapEditor(mpu::LogBuffer& buffer);

    void run(mpu::gph::Renderer2D& renderer);

private:
    void handleMainMenu();
    void handleSidebar();
    void setupInputs();
    void setupKeybindings();

    void setWorkdir(fs::path newWorkdir);
    void reloadAssets();

    // imgui bools
    bool m_showLogWindow=false;
    bool m_showPerfWindow=false;
    bool m_showRMWindow = false;
    bool m_showDemoWindow = false;
    bool m_showFPS = true;
    bool m_showGrid = true;

    mpu::gph::Camera2D cam;
    mpu::LogBuffer& m_logBuffer;
    fs::path m_workdir;

    // selection
    int m_selectedSprite{-1};
    int m_selectedTile{-1};

    // assets
    std::vector<metaSprite> m_sprites;
    std::vector<metaTile> m_tiles;
};


#endif //MPUTILS_TILEMAPEDITOR_H
