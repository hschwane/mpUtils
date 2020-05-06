/*
 * mpUtils
 * ImGuiWindows.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Gui/ImGuiWindows.h"
#include "mpUtils/Graphics/Gui/ImGuiStyles.h"
#include "mpUtils/Graphics/Utils/misc.h"
#include "mpUtils/Graphics/Input.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

void showStyleSelectorWindow(bool* show, bool drawAsChild)
{
    bool visible;
    if(drawAsChild)
    {
        visible = ImGui::BeginChild("Style Selector");
    }
    else
    {
        ImGui::SetNextWindowSize(ImVec2(2400,120),ImGuiCond_FirstUseEver);
        visible = ImGui::Begin("Style Selector", show);
    }

    if(visible)
    {
        static int selected=-1;
        if(ImGui::Combo("style",&selected,"ImGui Default Dark\0ImGui Default Light\0ImGui Default Classic\0"
                                       "Corporate Grey Flat\0Corporate Grey\0Darcula\0Pagghiu\0LightGreen\0"
                                       "Yet Another Dark Theme\0Gold and Black\0\0"))
        {
            switch(selected)
            {
                case 0: ImGui::StyleImGuiDefaultDark(); break;
                case 1: ImGui::StyleImGuiDefaultLight(); break;
                case 2: ImGui::StyleImGuiDefaultClassic(); break;
                case 3: ImGui::StyleCorporateGreyFlat(); break;
                case 4: ImGui::StyleCorporateGrey(); break;
                case 5: ImGui::StyleDarcula(); break;
                case 6: ImGui::StylePagghiu(); break;
                case 7: ImGui::StyleLightGreen(); break;
                case 8: ImGui::StyleYetAnotherDarktheme(); break;
                case 9: ImGui::StyleGoldAndBlack(); break;
                default: break;
            }
        }

        if(ImGui::CollapsingHeader("Style Editor"))
            ImGui::ShowStyleEditor();
    }

    if(drawAsChild)
        ImGui::EndChild();
    else
        ImGui::End();
}

void showFPSOverlay(int corner, ImVec2 distance)
{
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 wndPos = ImVec2((corner & 1) ? io.DisplaySize.x - distance.x : distance.x, (corner & 2) ? io.DisplaySize.y - distance.y : distance.y);
    ImVec2 pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
    ImGui::SetNextWindowPos(wndPos, ImGuiCond_Always, pivot);
    if (ImGui::Begin("FPS overlay", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration
                                            |ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings
                                            | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
    {
        // calculate average over last 5 frames
        static int averageIndex=0;
        static float averageData[5];
        static float average;
        if(averageIndex==5)
        {
            averageIndex = 0;
            average = 0.2f * (averageData[0] + averageData[1] + averageData[2]
                              + averageData[3] + averageData[4]);
        }
        averageData[averageIndex++] = 1.0f / mpu::gph::Input::deltaTime();

        ImGui::Text("FPS: %.1f", average);
    }
    ImGui::End();
}

void showBasicPerformanceWindow(bool* show, bool drawAsChild)
{
    bool visible;
    if(drawAsChild)
    {
        visible = ImGui::BeginChild("Performance");
    }
    else
    {
        ImGui::SetNextWindowSize(ImVec2(180,130),ImGuiCond_FirstUseEver);
        visible = ImGui::Begin("Performance", show);
    }

    if(visible)
    {
        // settings
        constexpr int plotSize = 100;
        constexpr int averageSize = 5;
        static bool holdPlot = false;

        // plotting data
        static std::vector<float> frametimes(plotSize);
        static int insertIndex = 1;
        if(insertIndex == plotSize)
            insertIndex = 0;

        // calculate average over last 5 frames
        static int averageIndex=0;
        static float averageData[5];
        if(averageIndex==averageSize)
        {
            averageIndex = 0;
            frametimes[insertIndex++] = 0.2f * (averageData[0] + averageData[1] + averageData[2]
                                                + averageData[3] + averageData[4]);
        }
        if(!holdPlot)
            averageData[averageIndex++] = mpu::gph::Input::deltaTime();

        ImGui::Text("Frametime: %f", mpu::gph::Input::deltaTime());
        ImGui::SameLine();
        ImGui::Text("FPS: %f", 1.0f / mpu::gph::Input::deltaTime());

        ImVec2 availSpace = ImGui::GetContentRegionAvail();
        availSpace.y -= ImGui::GetTextLineHeightWithSpacing();
        ImGui::PlotHistogram("plotFrametime", frametimes.data(), 100,insertIndex-1, nullptr, 0, FLT_MAX, availSpace);

        ImGui::Checkbox("hold plot",&holdPlot);
        ImGui::SameLine();

        static bool vsyncOverride;
        if(ImGui::Checkbox("Enable V-Sync (override)", &vsyncOverride))
            mpu::gph::enableVsync(vsyncOverride);
    }

    if(drawAsChild)
        ImGui::EndChild();
    else
        ImGui::End();
}

}}