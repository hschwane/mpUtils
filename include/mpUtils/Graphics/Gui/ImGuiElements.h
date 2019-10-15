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
}

#endif  // RAPTOR_IMGUIELEMENTS_H
