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

/**
 * @brief Splits current region in two parts and allows to move the splitter
 * @param split_vertically
 * @param thickness
 * @param size1
 * @param size2
 * @param min_size1
 * @param min_size2
 * @param splitter_long_axis_size
 * @param visible should the splitter be visible
 * @return
 */
bool Splitter(bool split_vertically, float thickness, float* size1, float* size2, float min_size1, float min_size2, float splitter_long_axis_size = -1.0f, bool visible = true);

/**
 * @brief returns true if previous item was hovered for at least time seconds
 */
bool IsItemHovered(ImGuiHoveredFlags flags, float time);

/**
 * @brief Toggleable button, changes state when clicked, returns true when state was changed
 * @param label the label
 * @param v value contains current state, will be changed when state changes
 * @param size size of the button
 */
bool ToggleButton(const char* label, bool* v, const ImVec2& size = ImVec2(0,0));

/**
 * @brief if enabled will automatically scroll to ScrollMaxY. Will disable if user manually scrolls.
 * @param enable should autoscroll be enabled
 * @param persistent a boolean that is persistent over multiples calls of the function from the same window. Pass
 *      different variables for different windows.
 */
void AutoscrollToBottom(bool* enable, bool* persistent);

}

#endif  // RAPTOR_IMGUIELEMENTS_H
