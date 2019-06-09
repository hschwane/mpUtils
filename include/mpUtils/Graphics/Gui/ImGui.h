/*
 * mpUtils
 * imGui.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_IMGUI_H
#define MPUTILS_IMGUI_H

// includes
//--------------------
#include "mpUtils/Graphics/Window.h"
#include "mpUtils/external/imgui/imgui.h"
#include "mpUtils/external/imgui/stdlib/imgui_stdlib.h"
//--------------------

// namespace
//--------------------
namespace ImGui { // we extend the imgui namespace inside mpu
//--------------------

    /**
     * @brief Creates an imgui instance. It will be destroyed automatically when the window is closed.
     *          The Window will also take care of drawing the gui and starting new frames.
     *          You can find additional imGui Styles in ImGuiStyles.h.
     * @param window the window to which you want to bind this ImGui
     */
    void create(mpu::gph::Window& window);

    /**
     * @brief Use this to change settings that can normally not be changed between frame begin and frame end.
     *          Pass a function or lambda. It will be called once, before the next frame begins.
     * @param settingFunction function that changes the settings
     */
    void changeSettings(std::function<void()> settingFunction);

    /**
     * @brief The Hover callback will be called whenever the mouse enter ImGui area (true) or exits it (false).
     *          This will overwrite the current callback.
     * @param callback the callback function you want to have called.
     */
    void setHoverCallback(const std::function<void(bool)>& callback);

    void destroy(); //!< destroy imgui. will be done automatically
    void show(); //!< show the imgui after it was hidden
    void hide(); //!< hide the imgui from the screen
    bool isVisible(); //!< check if imgui is currently visible
    void toogleVisibility(); //!< toogle the visibility of imgui

}

#endif //MPUTILS_IMGUI_H
