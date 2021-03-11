/*
 * raptor
 * ImGuiElements.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Window.h"
#include "mpUtils/Graphics/Gui/ImGuiElements.h"
#include <mpUtils/external/imgui/imgui_internal.h>
#include "mpUtils/Misc/pointPicking.h"
#include "mpUtils/Graphics/Input.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
//--------------------

// namespace
//--------------------
namespace ImGui{
//--------------------

bool BufferingBar(const char* label, float value,  const ImVec2& size_arg, const ImU32& bg_col, const ImU32& fg_col) {
  ImGuiWindow* window = GetCurrentWindow();
  if (window->SkipItems)
    return false;

  ImGuiContext& g = *GImGui;
  const ImGuiStyle& style = g.Style;
  const ImGuiID id = window->GetID(label);

  ImVec2 pos = window->DC.CursorPos;
  ImVec2 size = size_arg;
  size.x -= style.FramePadding.x * 2;

  const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
  ItemSize(bb, style.FramePadding.y);
  if (!ItemAdd(bb, id))
    return false;

  // Render
  const float circleStart = size.x * 0.7f;
  const float circleEnd = size.x;
  const float circleWidth = circleEnd - circleStart;

  window->DrawList->AddRectFilled(bb.Min, ImVec2(pos.x + circleStart, bb.Max.y), bg_col);
  window->DrawList->AddRectFilled(bb.Min, ImVec2(pos.x + circleStart*value, bb.Max.y), fg_col);

  const float t = g.Time;
  const float r = size.y / 2;
  const float speed = 1.5f;

  const float a = speed*0;
  const float b = speed*0.333f;
  const float c = speed*0.666f;

  const float o1 = (circleWidth+r) * (t+a - speed * (int)((t+a) / speed)) / speed;
  const float o2 = (circleWidth+r) * (t+b - speed * (int)((t+b) / speed)) / speed;
  const float o3 = (circleWidth+r) * (t+c - speed * (int)((t+c) / speed)) / speed;

  window->DrawList->AddCircleFilled(ImVec2(pos.x + circleEnd - o1, bb.Min.y + r), r, bg_col);
  window->DrawList->AddCircleFilled(ImVec2(pos.x + circleEnd - o2, bb.Min.y + r), r, bg_col);
  window->DrawList->AddCircleFilled(ImVec2(pos.x + circleEnd - o3, bb.Min.y + r), r, bg_col);
  return true;
}

bool Spinner(const char* label, float radius, int thickness, const ImU32& color) {
  ImGuiWindow* window = GetCurrentWindow();
  if (window->SkipItems)
    return false;

  ImGuiContext& g = *GImGui;
  const ImGuiStyle& style = g.Style;
  const ImGuiID id = window->GetID(label);

  ImVec2 pos = window->DC.CursorPos;
  ImVec2 size((radius )*2, (radius + style.FramePadding.y)*2);

  const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
  ItemSize(bb, style.FramePadding.y);
  if (!ItemAdd(bb, id))
    return false;

  // Render
  window->DrawList->PathClear();

  int num_segments = 30;
  int start = abs(ImSin(g.Time*1.8f)*(num_segments-5));

  const float a_min = IM_PI*2.0f * ((float)start) / (float)num_segments;
  const float a_max = IM_PI*2.0f * ((float)num_segments-3) / (float)num_segments;

  const ImVec2 centre = ImVec2(pos.x+radius, pos.y+radius+style.FramePadding.y);

  for (int i = 0; i < num_segments; i++) {
    const float a = a_min + ((float)i / (float)num_segments) * (a_max - a_min);
    window->DrawList->PathLineTo(ImVec2(centre.x + ImCos(a+g.Time*8) * radius,
                                        centre.y + ImSin(a+g.Time*8) * radius));
  }

  window->DrawList->PathStroke(color, false, thickness);
  return true;
}

void LoadingIndicatorCircle(const char* label, const float indicator_radius,
                                   const ImVec4& main_color, const ImVec4& backdrop_color,
                                   const int circle_count, const float speed) {
  ImGuiWindow* window = GetCurrentWindow();
  if (window->SkipItems) {
    return;
  }

  ImGuiContext& g = *GImGui;
  const ImGuiID id = window->GetID(label);

  const ImVec2 pos = window->DC.CursorPos;
  const float circle_radius = indicator_radius / 10.0f;
  const ImRect bb(pos, ImVec2(pos.x + indicator_radius * 2.0f,
                              pos.y + indicator_radius * 2.0f));
  ItemSize(bb, g.Style.FramePadding.y);
  if (!ItemAdd(bb, id)) {
    return;
  }
  const float t = g.Time;
  const auto degree_offset = 2.0f * IM_PI / circle_count;
  for (int i = 0; i < circle_count; ++i) {
    const auto x = indicator_radius * std::sin(degree_offset * i);
    const auto y = indicator_radius * std::cos(degree_offset * i);
    const auto growth = std::max(0.0f, std::sin(t * speed - i * degree_offset));
    ImVec4 color;
    color.x = main_color.x * growth + backdrop_color.x * (1.0f - growth);
    color.y = main_color.y * growth + backdrop_color.y * (1.0f - growth);
    color.z = main_color.z * growth + backdrop_color.z * (1.0f - growth);
    color.w = 1.0f;
    window->DrawList->AddCircleFilled(ImVec2(pos.x + indicator_radius + x,
                                             pos.y + indicator_radius - y),
                                      circle_radius + growth * circle_radius,
                                      GetColorU32(color));
  }
}

void SimpleModal(const std::string& header, std::string text, std::vector<std::string> buttons, std::string icon,
                 std::function<void(int)> callback)
{
    std::string uniqueHeader = header + "##" + std::to_string(mpu::getRanndomSeed());

    // calculate size of the dialog
    auto textSize = ImGui::CalcTextSize(text.c_str());
    float w2 = textSize.x * textSize.y * 4;
    float w = sqrt(w2) + 80;

    float buttonW = 0;
    for (auto & button : buttons)
        buttonW += ImGui::CalcTextSize(button.c_str()).x+20;

    w = glm::max(w,buttonW+50);

    std::shared_ptr<int> i = std::make_shared<int>();
    *i = ImGui::getAttatchedWindow().addFrameBeginCallback(
            [i,uniqueHeader{move(uniqueHeader)},text{move(text)},buttons{move(buttons)},
             icon{move(icon)},callback{move(callback)}, needOpen{true}, width{w}]() mutable
            {
                auto wndSize = ImGui::getAttatchedWindow().getSize();
                ImGui::SetNextWindowSize(ImVec2(width,0),ImGuiCond_Always);
                if(ImGui::BeginPopupModal(uniqueHeader.c_str(), nullptr, ImGuiWindowFlags_NoScrollbar))
                {
                    ImGui::BeginHorizontal("ht",ImVec2(ImGui::GetWindowSize().x,0));
                    ImGui::Spring();

                    ImGui::PushTextWrapPos(ImGui::GetCursorPosX()+width-80);
                    ImGui::Text("%s",text.c_str());
                    ImGui::PopTextWrapPos();

                    ImGui::Spring(0.2);

                    ICON_BEGIN();
                    ImGui::Text("%s",icon.c_str());
                    ICON_END();

                    ImGui::Spring();
                    ImGui::EndHorizontal();


                    ImGui::SetCursorPosY(ImGui::GetCursorPosY()+5);

                    ImGui::BeginHorizontal("hb",ImVec2(ImGui::GetWindowSize().x,0));
                    ImGui::Spring(0.5);
                    for (int j=0; j < buttons.size(); j++)
                    {
                        if(ImGui::Button(buttons[j].c_str()))
                        {
                            if(callback)
                                callback(j);
                            ImGui::CloseCurrentPopup();
                        }
                    }
                    ImGui::Spring(0.5);
                    ImGui::EndHorizontal();
                    ImGui::EndPopup();
                } else if(needOpen)
                {
                    ImGui::OpenPopup(uniqueHeader.c_str());
                    needOpen = false;
                } else {
                    ImGui::getAttatchedWindow().removeFrameBeginCallback(*i);
                }
            });
}

int SimpleBlockingModal(const std::string& header, std::string text, std::vector<std::string> buttons, std::string icon)
{
    // calculate size of the dialog
    auto textSize = ImGui::CalcTextSize(text.c_str());
    float w2 = textSize.x * textSize.y * 4;
    float w = sqrt(w2) + 80;
    float buttonW = 0;
    for (auto & button : buttons)
        buttonW += ImGui::CalcTextSize(button.c_str()).x+20;
    float width = glm::max(w,buttonW+50);

    float textHeight = ImGui::CalcTextSize(text.c_str(), nullptr,false,width-80).y;
    float maxButtonH = 0;
    for (auto & button : buttons)
        maxButtonH = glm::max(maxButtonH,ImGui::CalcTextSize(button.c_str()).y);
    float height = textHeight+maxButtonH+60;

    // generate unique header
    std::string uniqueHeader = header + "##" + std::to_string(mpu::getRanndomSeed());

    // copy imgui context
    auto context = ImGui::GetCurrentContext();
    auto font = ImGui::GetFont();
    context->IO.Fonts->Locked = false;
    ImGui::SetCurrentContext(ImGui::CreateContext(context->IO.Fonts));
    ImGui::GetCurrentContext()->FontStack = context->FontStack;
    ImGui::GetCurrentContext()->Style = context->Style;

    // shutdown rendering on old os window
    mpu::gph::Window& wnd = ImGui::getAttatchedWindow();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    // create new os window
    mpu::gph::Window::setWindowHint(GLFW_DECORATED, GLFW_FALSE);
    mpu::gph::Window::setWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    mpu::gph::Window::setWindowHint(GLFW_FOCUSED, GLFW_TRUE);
    mpu::gph::Window::setWindowHint(GLFW_FLOATING, GLFW_TRUE);
    mpu::gph::Window popup(width,height,"popup");
    popup.setPosition(wnd.getSize()/2 + wnd.getPosition() -popup.getSize()/2);
    popup.makeContextCurrent();

    // initialize imgui rendering on the new window
    ImGui_ImplGlfw_InitForOpenGL(popup.window(), false);
    ImGui_ImplOpenGL3_Init("#version 130");

    // popup window main loop
    int button = -1;
    bool openPopup = true;
    bool m_close = false;
    while (popup.frameBegin() && !m_close)
    {
        // begin the frame
        mpu::gph::Input::update();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetCurrentFont(font);

        // open popup on forst iteration
        if(openPopup)
        {
            ImGui::OpenPopup(uniqueHeader.c_str());
            openPopup = false;
        }

        // draw the actual modal dialog
        ImGui::SetNextWindowSize(ImVec2(width,height),ImGuiCond_Always);
        if(ImGui::BeginPopupModal(uniqueHeader.c_str(), nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize))
        {
            ImGui::BeginHorizontal("ht",ImVec2(ImGui::GetWindowSize().x,0));
            ImGui::Spring();

            ImGui::PushTextWrapPos(ImGui::GetCursorPosX()+width-80);
            ImGui::Text("%s",text.c_str());
            ImGui::PopTextWrapPos();

            ImGui::Spring(0.2);

            ICON_BEGIN();
            ImGui::Text("%s",icon.c_str());
            ICON_END();

            ImGui::Spring();
            ImGui::EndHorizontal();

            ImGui::SetCursorPosY(ImGui::GetCursorPosY()+5);

            ImGui::BeginHorizontal("hb",ImVec2(ImGui::GetWindowSize().x,0));
            ImGui::Spring(0.5);
            for (int j=0; j < buttons.size(); j++)
            {
                ImGui::SetCurrentFont(font);
                if(ImGui::Button(buttons[j].c_str()))
                {
                    button = j;
                    ImGui::CloseCurrentPopup();
                    m_close = true;
                }
            }
            ImGui::Spring(0.5);
            ImGui::EndHorizontal();
            ImGui::EndPopup();
        }

        ImGui::Render();
        glViewport(0, 0, popup.getSize().x, popup.getSize().y);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        popup.frameEnd();
    }

    // reset state
    mpu::gph::Window::resetWindowHints();

    // remove tempoary context
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // restore old context
    wnd.makeContextCurrent();
    ImGui::SetCurrentContext(context);
    ImGui_ImplGlfw_InitForOpenGL(wnd.window(),false);
    ImGui_ImplOpenGL3_Init("#version 130");
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    context->IO.Fonts->Locked = true;

    return button;
}

bool Splitter(bool split_vertically, float thickness, float* size1, float* size2, float min_size1, float min_size2, float splitter_long_axis_size, bool visible)
{
    using namespace ImGui;
    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = g.CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    bb.Min.x = window->DC.CursorPos.x + (split_vertically ? *size1 : 0.0f);
    bb.Min.y = window->DC.CursorPos.y + (split_vertically ? 0.0f : *size1);

    ImVec2 itemSize = CalcItemSize(split_vertically ? ImVec2(thickness, splitter_long_axis_size) : ImVec2(splitter_long_axis_size, thickness), 0.0f, 0.0f);
    bb.Max.x = bb.Min.x + itemSize.x;
    bb.Max.y = bb.Min.y + itemSize.y;

    ImGuiAxis axis = split_vertically ? ImGuiAxis_X : ImGuiAxis_Y;
    float hover_extend=1.0f;
    float hover_visibility_delay=0.f;

    // --------------------------------------------------------------
    // copied from  ImGui::SplitterBehavior and modified
    const ImGuiItemFlags item_flags_backup = window->DC.ItemFlags;
    window->DC.ItemFlags |= ImGuiItemFlags_NoNav | ImGuiItemFlags_NoNavDefaultFocus;
    bool item_add = ItemAdd(bb, id);
    window->DC.ItemFlags = item_flags_backup;
    if (!item_add)
        return false;

    bool hovered, held;
    ImRect bb_interact = bb;
    bb_interact.Expand(axis == ImGuiAxis_Y ? ImVec2(0.0f, hover_extend) : ImVec2(hover_extend, 0.0f));
    ButtonBehavior(bb_interact, id, &hovered, &held, ImGuiButtonFlags_FlattenChildren | ImGuiButtonFlags_AllowItemOverlap);
    if (g.ActiveId != id)
        SetItemAllowOverlap();

    if (held || (g.HoveredId == id && g.HoveredIdPreviousFrame == id && g.HoveredIdTimer >= hover_visibility_delay))
        SetMouseCursor(axis == ImGuiAxis_Y ? ImGuiMouseCursor_ResizeNS : ImGuiMouseCursor_ResizeEW);

    ImRect bb_render = bb;
    if (held)
    {
        ImVec2 mouse_delta_2d;
        mouse_delta_2d.x = g.IO.MousePos.x - g.ActiveIdClickOffset.x - bb_interact.Min.x;
        mouse_delta_2d.y = g.IO.MousePos.y - g.ActiveIdClickOffset.y - bb_interact.Min.y;
        float mouse_delta = (axis == ImGuiAxis_Y) ? mouse_delta_2d.y : mouse_delta_2d.x;

        // Minimum pane size
        float size_1_maximum_delta = ImMax(0.0f, *size1 - min_size1);
        float size_2_maximum_delta = ImMax(0.0f, *size2 - min_size2);
        if (mouse_delta < -size_1_maximum_delta)
            mouse_delta = -size_1_maximum_delta;
        if (mouse_delta > size_2_maximum_delta)
            mouse_delta = size_2_maximum_delta;

        // Apply resize
        if (mouse_delta != 0.0f)
        {
            if (mouse_delta < 0.0f)
                IM_ASSERT(*size1 + mouse_delta >= min_size1);
            if (mouse_delta > 0.0f)
                IM_ASSERT(*size2 - mouse_delta >= min_size2);
            *size1 += mouse_delta;
            *size2 -= mouse_delta;
            bb_render.Translate((axis == ImGuiAxis_X) ? ImVec2(mouse_delta, 0.0f) : ImVec2(0.0f, mouse_delta));
            MarkItemEdited(id);
        }
    }

    // Render
    if(visible)
    {
        const ImU32 col = GetColorU32(
                held ? ImGuiCol_SeparatorActive : (hovered && g.HoveredIdTimer >= hover_visibility_delay)
                                                  ? ImGuiCol_SeparatorHovered : ImGuiCol_Separator);
        window->DrawList->AddRectFilled(bb_render.Min, bb_render.Max, col, 0.0f);
    }

    return held;
}

bool IsItemHovered(ImGuiHoveredFlags flags, float time)
{
    return IsItemHovered(flags) && GImGui->HoveredIdTimer > time;
}

bool ToggleButton(const char* label, bool* v, const ImVec2& size)
{
    if(*v)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if(ImGui::Button(label,size))
        {
            *v = false;
            ImGui::PopStyleColor(2);
            return true;
        }
        ImGui::PopStyleColor(2);

    } else
    {
        if(ImGui::Button(label,size))
        {
            *v = true;
            return true;
        }
    }
    return false;
}

void AutoscrollToBottom(bool* enable, bool* persistent)
{
    if(*enable)
        ImGui::SetScrollY(ImGui::GetScrollMaxY());

    bool isScrollAtEnd = (ImGui::GetScrollY() == ImGui::GetScrollMaxY());
    if(isScrollAtEnd && !*persistent)
        *enable = true;
    if(!isScrollAtEnd && !*persistent)
        *enable = false;
    *persistent = isScrollAtEnd;
}

}