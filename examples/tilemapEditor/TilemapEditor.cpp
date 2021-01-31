/*
 * mpUtils
 * TilemapEditor.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the TilemapEditor class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "TilemapEditor.h"
//--------------------

// resource manager
ResourceManagerType& getRM()
{
    // pointer to resource caches to resolve recursive dependencies
    static mpu::ImageRC* imgrc = nullptr;
    static mpu::gph::Sprite2DRC* sprtrc = nullptr;

    // construct a resource manager
    static ResourceManagerType resourceManager(
            // add support for images
            {mpu::preloadImage,mpu::finalLoadImage,"",
             mpu::getDefaultImage(), "Image-8bit"},
            // add support for sprites
            {[&](const std::string& data){ return mpu::gph::preloadSprite2D(imgrc, data); },
             mpu::gph::finalLoadSprite2D,"",
             mpu::gph::getDefaultSprite(), "Sprite2D"}
    );

    // populate the pointers for recursive dependencies, only once
    [[maybe_unused]] static bool once = [&]()
    {
        imgrc = &resourceManager.get<mpu::Image8>();
        sprtrc = &resourceManager.get<mpu::gph::Sprite2D>();
        return true;
    } ();

    return resourceManager;
}
template class mpu::ResourceManager<mpu::ImageRC,mpu::gph::Sprite2DRC>;

// function definitions of the TilemapEditor class
//-------------------------------------------------------------------

TilemapEditor::TilemapEditor(mpu::LogBuffer& buffer)
    : m_logBuffer(buffer)
{
    setupInputs();
    setupKeybindings();

    // start resource manager
    getRM();
    setWorkdir(MAP_WORKDIR);

    // enable drag and drop
    mpu::gph::Input::addDropCallback([this](mpu::gph::Window& wnd, std::vector<std::string> files)
    {
        for(const auto& file : files) {
            fs::path p(file);

            if(fs::is_regular_file(p) && p.has_extension())
            {
                // compy image and create sprite
                if(p.extension() == ".png" || p.extension() == ".jpg" || p.extension() == ".jpeg" ||
                    p.extension() == ".bmp" || p.extension() == ".tga" || p.extension() == ".psd")
                {
                    // check if file exists and copy
                    fs::path target = "images/";
                    target += p.filename();
                    for(int i=0; fs::exists(target); ++i) {
                        target.replace_filename(std::string(target.stem()) + "_" + std::to_string(i)
                            + std::string(target.extension()));
                    }
                    fs::copy(p,target);

                    // create sprite
                    auto sd = mpu::gph::makeSimpleSprite(target);
                    toml::store( "sprites/" + std::string(target.stem()) + ".sprite", sd.toToml());

                }
                else if(p.extension() == ".sprite") {

                }
                else if(p.extension() == ".tile") {

                }
                else if(p.extension() == ".tilemap") {

                }
            }
        }
        reloadAssets();
    });
}

void TilemapEditor::run(mpu::gph::Renderer2D& renderer)
{
    // handle imGui
    handleMainMenu();
    handleSidebar();

    if(m_showLogWindow) mpu::gph::showLoggerWindow(m_logBuffer,&m_showLogWindow);
    if(m_showPerfWindow) mpu::gph::showBasicPerformanceWindow(&m_showPerfWindow);
    if(m_showRMWindow) mpu::gph::showResourceManagerDebugWindow(getRM(),&m_showRMWindow);
    if(m_showDemoWindow) ImGui::ShowDemoWindow(&m_showDemoWindow);
    if(m_showFPS) mpu::gph::showFPSOverlay(1, {5.0, 25.0});

    // handle map interaction

    // draw current map
}

void TilemapEditor::setWorkdir(fs::path newWorkdir)
{
    fs::current_path(newWorkdir);
    fs::create_directory("images");
    fs::create_directory("sprites");
    fs::create_directory("tilemaps");
    m_workdir = newWorkdir;
    reloadAssets();
}

void TilemapEditor::reloadAssets()
{
    // clear everything
    m_sprites.clear();
    m_tiles.clear();
    getRM().tryReleaseAll();

    // load sprites
    std::set<fs::path> spritePaths;
    for(fs::directory_entry const& entry : fs::directory_iterator("sprites")) {
        fs::path p = entry.path();
        if(fs::is_regular_file(p) && p.has_extension() && p.extension() == ".sprite")
            spritePaths.insert(p);
    }
    for(const auto& p : spritePaths) {
        metaSprite s;
        std::string content = mpu::readFile(p);
        s.filename = p.stem();
        s.data = mpu::gph::Sprite2DData(content);
        s.sprite = getRM().load<mpu::gph::Sprite2D>(p);
        m_sprites.push_back(s);
    }

    // load tiles

    // load maps

}

void TilemapEditor::setupInputs()
{
    using namespace mpu::gph;
    namespace ip = mpu::gph::Input;

    ip::addButton("Toggle Fullscreen","Switch between window and fullscreen.",[](Window& wnd){ wnd.toggleFullscreen();});
    ip::addButton("Toggle GUI","Hides / shows GUI.",[](Window& wnd){ ImGui::toggleVisibility();});

}

void TilemapEditor::setupKeybindings()
{
    using namespace mpu::gph;
    namespace ip = mpu::gph::Input;

    ip::mapKeyToInput("Toggle Fullscreen",GLFW_KEY_F11);
    ip::mapKeyToInput("Toggle GUI",GLFW_KEY_F10);

    ip::mapScrollToInput("Camera2DZoom");
    ip::mapKeyToInput("Camera2DMoveDownUp",GLFW_KEY_W,ip::ButtonBehavior::whenDown,ip::AxisBehavior::positive);
    ip::mapKeyToInput("Camera2DMoveDownUp",GLFW_KEY_S,ip::ButtonBehavior::whenDown,ip::AxisBehavior::negative);
    ip::mapKeyToInput("Camera2DMoveLeftRight",GLFW_KEY_D,ip::ButtonBehavior::whenDown,ip::AxisBehavior::positive);
    ip::mapKeyToInput("Camera2DMoveLeftRight",GLFW_KEY_A,ip::ButtonBehavior::whenDown,ip::AxisBehavior::negative);
}

void TilemapEditor::handleMainMenu()
{
    if(ImGui::BeginMainMenuBar()) {
        if(ImGui::BeginMenu("File")) {
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("View")) {
            if(ImGui::MenuItem("Toggle Fullscreen","F10"))
//                wnd.toggleFullscreen();
            ImGui::MenuItem("Show FPS", nullptr, &m_showFPS);
            ImGui::MenuItem("Show grid", nullptr, &m_showGrid);
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("Windows")) {
            ImGui::MenuItem("Log", nullptr, &m_showLogWindow);
            ImGui::MenuItem("Performance", nullptr, &m_showPerfWindow);
            ImGui::MenuItem("Resources", nullptr, &m_showRMWindow);
            ImGui::Separator();
            ImGui::MenuItem("ImGuiDemoWindow", nullptr, &m_showDemoWindow);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void TilemapEditor::handleSidebar()
{
    float previewSize = 16;
    float tooltipSize = 256;

    glm::ivec2 wndSize = ImGui::getAttatchedWindow().getSize();
    ImGui::SetNextWindowPos({0,20},ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints(ImVec2(128,wndSize.y-20),ImVec2(wndSize.x,wndSize.y-20) );
    if(ImGui::Begin("content", nullptr,  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar)) {

        float wndHeight = ImGui::GetWindowContentRegionMax().y;
        static float oldWndHeight = wndHeight;
        static float size1 = wndHeight*0.3f;
        static float size2 = wndHeight*0.3f;
        static float size3 = wndHeight*0.3f;
        if(wndHeight != oldWndHeight)
        {
            size1 = wndHeight * (size1 / oldWndHeight);
            size2 = wndHeight * (size1 / oldWndHeight);
            size3 = wndHeight * (size1 / oldWndHeight);
            oldWndHeight = wndHeight;
        }

        if(ImGui::CollapsingHeader("Sprites")) {
            for(int i = 0; i < m_sprites.size(); i++)
            {
                ImGui::PushID(i);
                if(ImGui::Selectable("##", m_selectedSprite == i, 0, ImVec2(0, previewSize)))
                    m_selectedSprite = i;

                if(ImGui::IsItemHovered(0,0.25))
                {
                    ImGui::BeginTooltip();
                    ImGui::Text("Sprite: %s",m_sprites[i].filename.c_str());
                    if(!m_sprites[i].data.spritesheet.empty())
                        ImGui::Text("Spritesheet: %s", m_sprites[i].data.spritesheet.c_str());

                    ImGui::Image((void*)(intptr_t)static_cast<GLuint>(m_sprites[i].sprite->getTexture()),
                                 ImVec2(tooltipSize, tooltipSize * m_sprites[i].sprite->getBaseTransform()[1][1] / m_sprites[i].sprite->getBaseTransform()[0][0]),
                                 ImVec2(0,1),ImVec2(1,0));

                    ImGui::Text("Semi-Transparancy: %s", m_sprites[i].data.semiTransparent ? "yes" : "no");
                    ImGui::Text("World size: %s", glm::to_string(m_sprites[i].data.worldSize).c_str());
                    ImGui::Text("Pivot: %s", glm::to_string(m_sprites[i].data.pivot).c_str());
                    ImGui::Text("ForwardDirecton: %f", mpu::deg(m_sprites[i].data.forward));
                    ImGui::Text("Tileing: %f", m_sprites[i].data.tileFactor);

                    ImGui::EndTooltip();
                }

                ImGui::SameLine();
                ImGui::Image((void*)(intptr_t)static_cast<GLuint>(m_sprites[i].sprite->getTexture()),
                             ImVec2(previewSize, previewSize * m_sprites[i].sprite->getBaseTransform()[1][1] / m_sprites[i].sprite->getBaseTransform()[0][0]),
                             ImVec2(0,1),ImVec2(1,0));
                ImGui::SameLine();
                ImGui::Text("%s", m_sprites[i].filename.c_str());
                ImGui::PopID();
            }
        }

        if(ImGui::CollapsingHeader("Tile types")) {

        }

        if(ImGui::CollapsingHeader("Maps")) {

        }

        float previewSize = glm::min(64.0f, ImGui::GetWindowContentRegionWidth() / 4);
//            for(int i = 0; i < m_activeTiles.size(); i++)
//            {
//                ImGui::PushID(i);
//                if(ImGui::Selectable("##", m_selectedTile == i, 0, ImVec2(0, previewSize)))
//                    m_selectedTile = i;
//                ImGui::SameLine();
//                ImGui::Image((void*)(intptr_t)static_cast<GLuint>(m_activeTiles[i].get()->getSprite(0).getTexture()),
//                             ImVec2(previewSize, previewSize));
//                ImGui::SameLine();
//                ImGui::Text("%s", m_activeTiles[i].get()->getName().c_str());
//                ImGui::PopID();
//            }
    }
    ImGui::End();
}
