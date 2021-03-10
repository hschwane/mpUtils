/*
 * mpUtils
 * ImGui.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Gui/ImGui.h"
#include "mpUtils/Graphics/Gui/ImGuiStyles.h"
#include <mpUtils/external/imgui/imgui_internal.h>
#include <mpUtils/paths.h>
#include <unordered_map>
#include "mpUtils/Log/Log.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "mpUtils/Graphics/Input.h"
#include "mpUtils/Misc/EmbeddedData.h"
//--------------------


// namespace
//--------------------
namespace ImGui {
//--------------------

ImFont* iconFont = nullptr;

namespace
{
    mpu::gph::Window* attatchedWindow = nullptr; //!< the window imGui is attatched to
    glm::ivec2 framebufferSize; //!< the current framebuffer size
    int closeCalbackId; //!< id of the window close callback
    int framebufferCallbackId; //!< id of the framebuffer resize callback
    int frameBeginCallbackId; //!< id of frame begin callback
    int frameEndCallbackId; //!< id of frame end callback
    bool visible=true; //!< should the gui be rendered?
    bool locked=false; //!< should the gui be rendered?
    float oldAlpha; //!< stores the old alpha when locking
    bool shouldDestroy = false; //!< should this instance be destroyed after ending the frame?
    bool captureMouseLastFrame=false; //!< did imGui capture the mouse last frame?
    std::vector<std::function<void()>> settingFunctions; //!< vector of setting change functions that need to be executed before the next FrameBegin
    std::function<void(bool)> guiHoverCallback; //!< callback called whenever mouse starts to hover the gui (true) or leaves the gui (false)

    std::unordered_map<std::string, ImFont*> fonts; //!< loaded imgui fonts are stored here

    void destroyInternal()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        attatchedWindow = nullptr;
    }
}


void create(mpu::gph::Window& window)
{
    assert_critical(attatchedWindow== nullptr, "ImGui", "There can only be one gui context.")
    assert_critical(IMGUI_CHECKVERSION(), "ImGui", "Im gui version mismatch.")
    ImGui::CreateContext();
    assert_critical( ImGui_ImplGlfw_InitForOpenGL(window.window(), true), "ImGui", "Failed to initialize imgui GLFW implementation." )
    assert_critical( ImGui_ImplOpenGL3_Init("#version 130"), "ImGui", "Failed to initialize imgu OpenGL implementation")

    framebufferSize = window.getFramebufferSize();

    // store the window and make sure we destroy the gui when the window gets destroyed
    attatchedWindow = &window;
    closeCalbackId = window.addCloseCallback([](){shouldDestroy = true;});

    // add resize callback
    framebufferCallbackId = window.addFBSizeCallback([](int w, int h){ framebufferSize.x=w; framebufferSize.y=h;});

    // begin frame callback, so this is done automatically
    frameBeginCallbackId = window.addFrameBeginCallback( [&window]()
    {
        if(shouldDestroy)
        {
            destroyInternal();
            shouldDestroy=false;
            return;
        }

        // call setings functions
        for(const auto &function : settingFunctions)
        {
            function();
        }
        settingFunctions.clear();

        // begin the frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        bool captureMouse = ImGui::GetIO().WantCaptureMouse;
        bool disableKeyInput = ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantTextInput;
        bool disableCursorInput = captureMouse && ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow);

        if(disableKeyInput && mpu::gph::Input::isKeyboardInputEnabled())
            mpu::gph::Input::disableKeyboardInput();
        else if(!disableKeyInput && !mpu::gph::Input::isKeyboardInputEnabled())
            mpu::gph::Input::enableKeyboardInput();

        if(disableCursorInput && mpu::gph::Input::isCursourInputEnabled())
            mpu::gph::Input::disableCursourInput();
        else if(!disableCursorInput && !mpu::gph::Input::isCursourInputEnabled())
            mpu::gph::Input::enableCursourInput();

        if( captureMouse != captureMouseLastFrame)
        {
            if(!captureMouse)
            {
                mpu::gph::Input::enableMouseInput();
                window.restoreCursor();
                if(guiHoverCallback)
                    guiHoverCallback(true);
            } else
            {
                mpu::gph::Input::disableMouseInput();
                if(guiHoverCallback)
                    guiHoverCallback(false);
            }

            captureMouseLastFrame = captureMouse;
        }
    });

    // end frame callback where gui is rendered
    frameEndCallbackId = window.addFrameEndCallback([]()
    {
        if(ImGui::GetCurrentContext()->WithinFrameScope)
        {
            if(visible)
            {
                ImGui::Render();
                glViewport(0, 0, framebufferSize.x, framebufferSize.y);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            } else
            {
                ImGui::EndFrame();
            }
        }
    });


    ImGui::GetIO().Fonts->TexDesiredWidth = 1024;

    // load icons
    static const ImWchar icons_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
    ImFontConfig icons_config;
    icons_config.PixelSnapH = true;
    icons_config.GlyphMinAdvanceX = 26;
    iconFont = ImGui::GetIO().Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH "fonts/fontawesome-webfont.ttf",26,&icons_config,icons_ranges);

    ImGui::StyleDarcula();
    logDEBUG("ImGui") << "ImGui initialized!";
}

void changeSettings( std::function<void()> settingFunction)
{
    settingFunctions.push_back(std::move(settingFunction));
}

void destroy()
{
    assert_true(attatchedWindow != nullptr, "ImGui", "Destroying gui that was not even created");
    attatchedWindow->removeCloseCallback(closeCalbackId);
    attatchedWindow->removeFBSizeCallback(framebufferCallbackId);
    attatchedWindow->removeFrameBeginCallback(frameBeginCallbackId);
    attatchedWindow->removeFrameEndCallback(frameEndCallbackId);
    destroyInternal();
}

void setHoverCallback(const std::function<void(bool)>& callback)
{
    guiHoverCallback=callback;
}

void show()
{
    visible =true;
    unlock();
    logDEBUG("ImGui") << "ImGui visible!";
}

void hide()
{
    visible =false;
    lock();
    logDEBUG("ImGui") << "ImGui hidden!";
}

bool isVisible()
{
    return visible;
}

void toggleVisibility()
{
    if(isVisible())
        hide();
    else
        show();
}

void unlock()
{
    if(locked)
    {
        changeSettings([](){ ImGui::GetStyle().Alpha = oldAlpha; });
        ImGui_ImplGlfw_DisableInput(false);
        locked = false;
        logDEBUG("ImGui") << "ImGui unlocked!";
    }
}

void lock()
{
    if(!locked)
    {
        changeSettings([](){ oldAlpha = ImGui::GetStyle().Alpha;ImGui::GetStyle().Alpha=oldAlpha*0.5;});
        ImGui_ImplGlfw_DisableInput(true);
        locked = true;
        logDEBUG("ImGui") << "ImGui locked!";
    }
}

bool isLocked()
{
    return locked;
}

void toggleLock()
{
    if(isLocked())
        unlock();
    else
        lock();
}

mpu::gph::Window& getAttatchedWindow()
{
    return *attatchedWindow;
}

ImFont* loadFont(std::string file, float size, bool addIcons)
{
    auto io = ImGui::GetIO();

    std::string mapKey = file + "_size_" + std::to_string(size) + ((addIcons) ? "_wicons" : "");

    auto it = fonts.find(mapKey);
    if(it == fonts.end())
    {
        auto ret = fonts.emplace(mapKey,io.Fonts->AddFontFromFileTTF(file.c_str(),size));
        it = ret.first;

        if(addIcons)
        {
            static const ImWchar icons_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
            ImFontConfig icons_config;
            icons_config.MergeMode = true;
            icons_config.PixelSnapH = true;
            icons_config.GlyphMinAdvanceX = size;
            io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH "fonts/fontawesome-webfont.ttf",size,&icons_config,icons_ranges);
        }
    }

    return it->second;
}

void pushDisabled()
{
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.6f);
}

void popDisabled()
{
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
}

}