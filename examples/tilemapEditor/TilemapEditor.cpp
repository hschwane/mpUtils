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
        m_selectionByFilename.clear();
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
                        target.replace_filename(std::string(target.stem()) + "_copy" + std::string(target.extension()));
                    }
                    fs::copy(p,target);

                    // create sprite
                    auto sd = mpu::gph::makeSimpleSprite(target);

                    // check if file exists
                    fs::path sprf = "sprites/" + std::string(target.stem()) + ".sprite";
                    for(int i=0; fs::exists(sprf); ++i) {
                        sprf.replace_filename(std::string(sprf.stem()) + "_copy" + std::string(sprf.extension()));
                    }
                    toml::store( sprf, sd.toToml());
                    m_selectionByFilename.insert(sprf);
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
        applySelectionsfromFilenameList();
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
        s.filename = p;
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
    ip::addButton("Remove","Removes selected items.",[this](Window& wnd){removeSelected();});
    ip::addButton("Duplicate","Duplicate selected items.",[this](Window& wnd){duplicateSelected();});
}

void TilemapEditor::setupKeybindings()
{
    using namespace mpu::gph;
    namespace ip = mpu::gph::Input;

    ip::mapKeyToInput("Toggle Fullscreen",GLFW_KEY_F11);
    ip::mapKeyToInput("Toggle GUI",GLFW_KEY_F10);
    ip::mapKeyToInput("Remove",GLFW_KEY_DELETE);
    ip::mapKeyToInput("Duplicate",GLFW_KEY_D,ip::ButtonBehavior::onPress, ip::AxisBehavior::positive, GLFW_MOD_CONTROL);

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
        if(ImGui::BeginMenu("Edit")) {
            if(ImGui::MenuItem("Remove","Del"))
                removeSelected();
            if(ImGui::MenuItem("Duplicate","Ctrl+D"))
                duplicateSelected();
            ImGui::Separator();
            if(ImGui::MenuItem("Edit Sprite","",false,m_selectedSprites.size()==1))
                editSprite(*m_selectedSprites.begin());
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("View")) {
            if(ImGui::MenuItem("Toggle Fullscreen","F10"))
                ImGui::getAttatchedWindow().toggleFullscreen();
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

        ImGui::SetNextItemOpen(true,ImGuiCond_Once);
        if(!m_selectedSprites.empty())
            ImGui::SetNextItemOpen(true);
        if(ImGui::CollapsingHeader("Sprites")) {
            ImGui::PushID("sprites");
            for(int i = 0; i < m_sprites.size(); i++)
            {
                ImGui::PushID(i);
                if(ImGui::Selectable("##", isSelected(i,m_selectedSprites), ImGuiSelectableFlags_AllowDoubleClick,
                                     ImVec2(0, previewSize))) {
                    select(i, m_selectedSprites);
                    if(ImGui::IsMouseDoubleClicked(0))
                        editSprite(i);
                }

                if(ImGui::IsItemHovered(0,0.25))
                {
                    ImGui::BeginTooltip();
                    ImGui::Text("Sprite: %s",m_sprites[i].filename.stem().c_str());
                    if(!m_sprites[i].data.spritesheet.empty())
                        ImGui::Text("Spritesheet: %s", m_sprites[i].data.spritesheet.c_str());

                    ImGui::Image((void*)(intptr_t)static_cast<GLuint>(m_sprites[i].sprite->getTexture()),
                                 ImVec2(tooltipSize, tooltipSize * m_sprites[i].sprite->getBaseTransform()[1][1] / m_sprites[i].sprite->getBaseTransform()[0][0]),
                                 ImVec2(0,1),ImVec2(1,0));

                    ImGui::BulletText("Semi-Transparancy: %s", m_sprites[i].data.semiTransparent ? "yes" : "no");
                    ImGui::BulletText("World size: %s", glm::to_string(m_sprites[i].data.worldSize).c_str());
                    ImGui::BulletText("Pivot: %s", glm::to_string(m_sprites[i].data.pivot).c_str());
                    ImGui::BulletText("ForwardDirecton: %f", mpu::deg(m_sprites[i].data.forward));
                    ImGui::BulletText("Tileing: %f", m_sprites[i].data.tileFactor);

                    ImGui::EndTooltip();
                }

                if(ImGui::BeginPopupContextItem())
                {
                    select(i,m_selectedSprites);
                    if(ImGui::MenuItem("Edit"))
                        editSprite(i);
                    if(ImGui::MenuItem("Remove"))
                        removeSelected();
                    if(ImGui::MenuItem("Duplicate"))
                        duplicateSelected();

                    ImGui::EndPopup();
                }

                ImGui::SameLine();
                ImGui::Image((void*)(intptr_t)static_cast<GLuint>(m_sprites[i].sprite->getTexture()),
                             ImVec2(previewSize, previewSize * m_sprites[i].sprite->getBaseTransform()[1][1] / m_sprites[i].sprite->getBaseTransform()[0][0]),
                             ImVec2(0,1),ImVec2(1,0));
                ImGui::SameLine();
                ImGui::Text("%s", m_sprites[i].filename.stem().c_str());
                ImGui::PopID();
            }
            ImGui::PopID();
        } else {
            m_selectedSprites.clear();
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

void TilemapEditor::select(int item, std::set<int>& selectedSet)
{
    if(ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_SHIFT)) {
        int front = *selectedSet.begin();
        if(item < front)
            while(item < front) {
                selectedSet.insert(item);
                ++item;
            }
        else
            while(item > front) {
                selectedSet.insert(item);
                --item;
            }
    } else if(ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL) || ImGui::IsKeyDown(GLFW_KEY_RIGHT_CONTROL)) {
        auto iter = std::find(selectedSet.begin(), selectedSet.end(), item);
        if(iter == selectedSet.end())
            selectedSet.insert(item);
        else
            selectedSet.erase(iter);
    } else {
        selectedSet.clear();
        selectedSet.insert(item);
    }
}

bool TilemapEditor::isSelected(int item, std::set<int>& selectedSet)
{
    auto iter = selectedSet.find(item);
    return iter != selectedSet.end();
}

void TilemapEditor::duplicateSelected()
{
    m_selectionByFilename.clear();
    for(int i: m_selectedSprites) {

        // check if file exists
        fs::path target = m_sprites[i].filename;
        for(int i=0; fs::exists(target); ++i) {
            target.replace_filename(std::string(target.stem()) + "_copy" + std::string(target.extension()));
        }

        // copy
        fs::copy(m_sprites[i].filename, target);

        // prepare selection
        m_selectionByFilename.insert(target);
    }

    reloadAssets();
    applySelectionsfromFilenameList();
}

void TilemapEditor::removeSelected()
{
    if(m_selectedSprites.empty()) return;
    ImGui::SimpleModal("Remove items?","Delete selected items from disk?\n(also deletes images in workdir)",
                       {"Yes","No"}, ICON_FA_EXCLAMATION_TRIANGLE, [this](int b){
        if(b == 0)
        {
            // remove sprite files
            std::vector<std::string> imgFilesToRemove;
            for(int i: m_selectedSprites) {
                fs::path sf = m_sprites[i].filename;
                logINFO("TilemapEditor") << "Removing file " << sf;
                fs::remove(sf);

                m_sprites[i].filename = "";
                imgFilesToRemove.push_back(m_sprites[i].data.texture);
            }

            // remove sprites from internal list
            m_sprites.erase( remove_if(m_sprites.begin(), m_sprites.end(), [](metaSprite& x){
                return x.filename.empty();
            }), m_sprites.end() );
            m_selectedSprites.clear();

            // remove image files
            for(const auto& imgf : imgFilesToRemove) {
                bool isInUse = false;
                for(const auto& sprite : m_sprites) {
                    if(sprite.data.texture == imgf)
                        isInUse = true;
                }
                if(!isInUse) {
                    logINFO("TilemapEditor") << "Removing file " << imgf;
                    fs::remove(imgf);
                }
            }

        }
    });
}

void TilemapEditor::applySelectionsfromFilenameList()
{
    m_selectedSprites.clear();
    for(int i=0; i < m_sprites.size(); ++i) {
        if(m_selectionByFilename.find(m_sprites[i].filename) != m_selectionByFilename.end())
            m_selectedSprites.insert(i);
    }
}

void TilemapEditor::editSprite(int i)
{
    logINFO("TilemapEditor") << "edit sprite";
}
