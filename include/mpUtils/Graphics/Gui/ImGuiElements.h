/*
 * raptor
 * ImGuiElements.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef RAPTOR_IMGUIELEMENTS_H
#define RAPTOR_IMGUIELEMENTS_H

// includes
//--------------------
#include "mpUtils/Graphics/Gui/ImGui.h"
#include <mpUtils/external/imgui/imgui_internal.h>
//--------------------

namespace ImGui {

// loading indicator from https://github.com/ocornut/imgui/issues/1901
bool BufferingBar(const char* label, float value,  const ImVec2& size_arg, const ImU32& bg_col, const ImU32& fg_col);
bool Spinner(const char* label, float radius, int thickness, const ImU32& color);
void LoadingIndicatorCircle(const char* label, const float indicator_radius,
                                   const ImVec4& main_color, const ImVec4& backdrop_color,
                                   const int circle_count, const float speed);

/**
 * @brief opens a modal popup, software continues to run underneath but user input is blocked
 *          each string in buttons creates a button. When when a button is clicked the modal will be closed
 *          and callback is executed with the id of the button pressed.
 */
void SimpleModal(const std::string& header, std::string text, std::vector<std::string> buttons,
        std::string icon="", std::function<void(int)> callback = {});

/**
 * @brief opens a modal popup, in a new os window. This wil interrupt fullscreen mode and
 *      be much slower then the asycronus simple dialog above. If a button is pressed The functions
 *      returns with the id of that button.
 */
int SimpleBlockingModal(const std::string& header, std::string text, std::vector<std::string> buttons,
                 std::string icon="");
}

#endif  // RAPTOR_IMGUIELEMENTS_H
