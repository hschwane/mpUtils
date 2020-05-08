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
#include <mpUtils/external/imgui/imgui_internal.h>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

void showLoggerWindow(LogBuffer& buffer, bool* show, bool drawAsChild)
{
    bool visible;
    if(drawAsChild)
    {
        visible = ImGui::BeginChild("Log");
    }
    else
    {
        ImGui::SetNextWindowSize(ImVec2(700,360),ImGuiCond_FirstUseEver);
        visible = ImGui::Begin("Log", show);
    }

    if(visible)
    {
        auto logLevelToColor = [](LogLvl lvl)
        {
            switch(lvl)
            {
                case LogLvl::FATAL_ERROR:
                    return ImVec4(0.95f,0.2f,0.2f,1.0f);
                case LogLvl::ERROR:
                    return ImVec4(0.95f,0.35f,0.35f,1.0f);
                case LogLvl::WARNING:
                    return ImVec4(0.92f,0.88f,0.53f,0.9f);
                case LogLvl::INFO:
                    return ImVec4(0.574f,0.93f,0.3534f,0.9f);
                case LogLvl::DEBUG:
                    return ImVec4(0.7661f,0.470f,0.84f,0.9f);
                case LogLvl::DEBUG2:
                    return ImVec4(0.7675f,0.5576f,0.82f,0.9f);
                default:
                    return ImVec4(1.0f,1.0f,1.0f,1.0f);
            }
        };

        // debug stuff
        if(ImGui::BeginPopupContextWindow("debug context popup"))
        {
            if(ImGui::MenuItem("Add 6 dummy entries"))
            {
                for(int i = 0; i < 1; i++)
                {
                    logDEBUG2("Test") << "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit...";
                    logDEBUG("Test") << "Morbi quam";
                    logINFO("Test") << "Maecenas quis eros fringilla.";
                    logWARNING("Test") << "Mauris felis magna, porta sit amet massa vitae, hendrerit mattis.";
                    logERROR("Test") << " Lorem ipsum dolor sit amet.";
                    logFATAL_ERROR("Test") << "Quisque eu est eget ipsum facilisis.";
                }
            }

            if(ImGui::MenuItem("Add 60 dummy entries"))
            {
                for(int i = 0; i < 10; i++)
                {
                    logDEBUG2("Test") << "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit...";
                    logDEBUG("Test") << "Morbi quam";
                    logINFO("Test") << "Maecenas quis eros fringilla.";
                    logWARNING("Test") << "Mauris felis magna, porta sit amet massa vitae, hendrerit mattis.";
                    logERROR("Test") << " Lorem ipsum dolor sit amet.";
                    logFATAL_ERROR("Test") << "Quisque eu est eget ipsum facilisis.";
                }
            }

            if(ImGui::MenuItem("Add 600 dummy entries"))
            {
                for(int i = 0; i < 100; i++)
                {
                    logDEBUG2("Test") << "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit...";
                    logDEBUG("Test") << "Morbi quam";
                    logINFO("Test") << "Maecenas quis eros fringilla.";
                    logWARNING("Test") << "Mauris felis magna, porta sit amet massa vitae, hendrerit mattis.";
                    logERROR("Test") << " Lorem ipsum dolor sit amet.";
                    logFATAL_ERROR("Test") << "Quisque eu est eget ipsum facilisis.";
                }
            }

            if(ImGui::MenuItem("Add 6000 dummy entries"))
            {
                for(int i = 0; i < 1000; i++)
                {
                    logDEBUG2("Test") << "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit...";
                    logDEBUG("Test") << "Morbi quam";
                    logINFO("Test") << "Maecenas quis eros fringilla.";
                    logWARNING("Test") << "Mauris felis magna, porta sit amet massa vitae, hendrerit mattis.";
                    logERROR("Test") << " Lorem ipsum dolor sit amet.";
                    logFATAL_ERROR("Test") << "Quisque eu est eget ipsum facilisis.";
                }
            }
            ImGui::EndPopup();
        }

        // filter
        static std::string messageFilter;
        static std::string moduleFilter;
        static std::string fileFilter;
        static std::array<bool,7> allowedLogLevels = {true,true,true,true,true,true,true};
        static std::thread::id threadFilter;

        ImGui::Button(ICON_FA_FILTER);
        if(ImGui::BeginPopupContextItem("filter by level",0))
        {
            ImGui::PushItemFlag(ImGuiItemFlags_SelectableDontClosePopup, true);
            if(ImGui::MenuItem("Enable all"))
            {
                allowedLogLevels = {true,true,true,true,true,true,true};
                buffer.setAllowedLogLevels(allowedLogLevels);
            }

            if(ImGui::MenuItem("Disable all"))
            {
                allowedLogLevels = {false,false,false,false,false,false,false};
                buffer.setAllowedLogLevels(allowedLogLevels);
            }

            ImGui::Separator();

            bool allowedLevelChanged = false;
            allowedLevelChanged |= ImGui::MenuItem("Other",nullptr, &allowedLogLevels[0]);
            allowedLevelChanged |= ImGui::MenuItem("Debug2",nullptr, &allowedLogLevels[6]);
            allowedLevelChanged |= ImGui::MenuItem("Debug",nullptr, &allowedLogLevels[5]);
            allowedLevelChanged |= ImGui::MenuItem("Info",nullptr, &allowedLogLevels[4]);
            allowedLevelChanged |= ImGui::MenuItem("Warning",nullptr, &allowedLogLevels[3]);
            allowedLevelChanged |= ImGui::MenuItem("Error",nullptr, &allowedLogLevels[2]);
            allowedLevelChanged |= ImGui::MenuItem("Fatal",nullptr, &allowedLogLevels[1]);
            if(allowedLevelChanged)
                buffer.setAllowedLogLevels(allowedLogLevels);

            ImGui::PopItemFlag();
            ImGui::EndPopup();
        }
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("filter by level");

        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        if(ImGui::InputText("module",&moduleFilter))
            buffer.setModuleFilter(moduleFilter);
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("Filter by module (\"-\" to exclude)");

        ImGui::SameLine();
        ImGui::SetNextItemWidth(160);
        if(ImGui::InputText("text",&messageFilter))
            buffer.setMessageFilter(messageFilter);
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("Filter by message (\"-\" to exclude)");

        ImGui::SameLine();
        ImGui::Button(ICON_FA_SLIDERS);
        if(ImGui::BeginPopupContextItem("advanced filter",0))
        {
            std::ostringstream ss;
            ss << std::setbase(16) << threadFilter;
            if(ss.str() == "thread::id of a non-executing thread")
                ss.str("All");
            ImGui::Text("Only showing Thread: %s",ss.str().c_str());

            if(ImGui::Button("Show all threads"))
            {
                threadFilter = std::thread::id();
                buffer.setThreadFilter(threadFilter);
            }

            ImGui::Text("Only show File:");
            if(ImGui::InputText("##filfilter",&fileFilter))
                buffer.setFileFilter(fileFilter);

            ImGui::EndPopup();
        }
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("Advanced filter options");

        ImGui::SameLine();
        if(ImGui::Button(ICON_FA_TIMES))
        {
            allowedLogLevels = {true,true,true,true,true,true,true};
            messageFilter = "";
            moduleFilter = "";
            fileFilter = "";
            threadFilter = std::thread::id();
            buffer.setMessageFilter(messageFilter);
            buffer.setModuleFilter(moduleFilter);
            buffer.setFileFilter(moduleFilter);
            buffer.setAllowedLogLevels(allowedLogLevels);
            buffer.setThreadFilter(threadFilter);
        }
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("Clear filter");

        // num of lines
        ImGui::SameLine();
        ImGui::Text("%i / %i /", buffer.filteredSize(), buffer.size());
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("num shown / num total / buffer capacity");

        // buffer capacity
        ImGui::SameLine();
        static int cap = buffer.capacity();
        ImGui::SetNextItemWidth(60);
        ImGui::DragInt("##",&cap,0.5,10,1000000);
        if(ImGui::IsItemDeactivatedAfterEdit())
            buffer.changeCapacity(cap);
        else if(!ImGui::IsItemActive())
            cap = buffer.capacity();
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("change buffer capacity");

        // copy to clipboard
        static bool copyFilename = false;
        ImGui::SameLine();
        if(ImGui::Button(ICON_FA_CLIPBOARD))
        {
            std::ostringstream clipboard;

            for(int i=0; i < buffer.filteredSize(); i++)
            {
                LogMessage& msg = buffer.filtered(i);

                struct tm timeStruct;
                #ifdef __linux__
                localtime_r(&msg.timepoint, &timeStruct);
                #elif _WIN32
                localtime_s(&timeStruct, &msg.timepoint);
                #else
                    #error please implement this for your operating system
                #endif

                if(msg.plaintext)
                {
                    if(msg.plaintext)
                        clipboard << msg.sMessage << std::endl;
                } else
                {
                    clipboard << "[" << toString(msg.lvl) << "]"
                              << " [" << std::put_time(&timeStruct, "%c") << "]";

                    if(!msg.sModule.empty())
                        clipboard << " (" << msg.sModule << "):";

                    clipboard << "\t" << msg.sMessage
                              << "\tThread: " << std::setbase(16) << msg.threadId << std::setbase(10);

                    if(copyFilename && !msg.sFilePosition.empty())
                        clipboard << "\t@File: " << msg.sFilePosition;

                    clipboard << std::endl;
                }
            }
            ImGui::SetClipboardText(clipboard.str().c_str());
        }
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("copy log to clipboard\nRight click for options.");
        if(ImGui::BeginPopupContextItem())
        {
            ImGui::PushItemFlag(ImGuiItemFlags_SelectableDontClosePopup, true);
            ImGui::MenuItem("enable copy filenames",nullptr,&copyFilename);
            ImGui::PopItemFlag();
            ImGui::EndPopup();
        }


        // clear buffer
        ImGui::SameLine();
        if(ImGui::Button(ICON_FA_TRASH_O))
            ImGui::SimpleModal("Clear logs","Sure you want to clear logs?",{"Yes","Cancel"},"",[&buffer](int i){ if(i==0) buffer.clear();});
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("clear logs");

        // autoscroll
        static bool autoscroll=true;
        ImGui::SameLine();
        ImGui::ToggleButton(ICON_FA_ARROW_DOWN,&autoscroll);
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("Scroll to end.");

        ImGui::Separator();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(0,0));
        ImGui::BeginChild("scroll");
        {
            static float lastScrollWndWidth = ImGui::GetWindowWidth();
            float scrollWndWidth = ImGui::GetWindowWidth();

            // setup columns
            ImGui::Columns(3,nullptr,true);
            static bool setColWidth=true;
            if(scrollWndWidth != lastScrollWndWidth)
            {
                lastScrollWndWidth = scrollWndWidth;
                setColWidth = true;
            }
            if(setColWidth)
            {
                ImGui::SetColumnWidth(0, 110);
                ImGui::SetColumnWidth(1, 120);
                setColWidth = false;
            }

            ImGuiListClipper clipper(buffer.filteredSize());
            while(clipper.Step())
                for(int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++)
                {
                    ImGui::PushID(i);
                    LogMessage& msg = buffer.filtered(i);

                    // setup invisible selectable to highlight line and show tooltip
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,ImVec4(0.45f,0.45f,0.45f,0.25f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive,ImVec4(0.45f,0.45f,0.45f,0.25f));
                    ImGui::Selectable("",false,ImGuiSelectableFlags_SpanAllColumns|ImGuiSelectableFlags_AllowItemOverlap);
                    ImGui::PopStyleColor(2);
                    if(ImGui::IsItemHovered(0,0.25f))
                    {
                        struct tm timeStruct;
                        #ifdef __linux__
                            localtime_r(&msg.timepoint, &timeStruct);
                        #elif _WIN32
                            localtime_s(&timeStruct, &msg.timepoint);
                        #else
                            #error "please implement this for your operating system"
                        #endif

                        ImGui::BeginTooltip();

                        std::ostringstream ss;
                        ss << std::put_time(&timeStruct, "%x %X");
                        ImGui::Text("Time: [%s]", ss.str().c_str());
                        ss.str("");
                        ss << std::setbase(16) << msg.threadId;
                        ImGui::Text("Thread: %s",ss.str().c_str());
                        ImGui::PushTextWrapPos(scrollWndWidth);
                        ImGui::TextWrapped("File: %s",msg.sFilePosition.c_str());
                        ImGui::Text("Right click for options.");
                        ImGui::PopTextWrapPos();
                        ImGui::EndTooltip();
                    }
                    if(ImGui::BeginPopupContextItem("popup"))
                    {
                        if(ImGui::MenuItem("Show only this Module"))
                        {
                            moduleFilter = msg.sModule;
                            buffer.setModuleFilter(moduleFilter);
                            ImGui::CloseCurrentPopup();
                        }
                        if(ImGui::MenuItem("Show only this Thread"))
                        {
                            threadFilter = msg.threadId;
                            buffer.setThreadFilter(threadFilter);
                            ImGui::CloseCurrentPopup();
                        }
                        if(ImGui::MenuItem("Show only this File"))
                        {
                            auto p = msg.sFilePosition.find(' ');
                            fileFilter = msg.sFilePosition.substr(0,p);
                            buffer.setFileFilter(fileFilter);
                            ImGui::CloseCurrentPopup();
                        }
                        ImGui::Separator();
                        if(ImGui::MenuItem("Copy to clipboard"))
                        {
                            std::ostringstream clipboard;

                            struct tm timeStruct;
                            #ifdef __linux__
                                localtime_r(&msg.timepoint, &timeStruct);
                            #elif _WIN32
                                localtime_s(&timeStruct, &msg.timepoint);
                            #else
                                #error please implement this for your operating system
                            #endif

                            if(msg.plaintext)
                            {
                                if(msg.plaintext)
                                    clipboard << msg.sMessage << std::endl;
                            }
                            else
                            {
                                clipboard <<  "[" << toString(msg.lvl) << "]"
                                     << " [" << std::put_time( &timeStruct, "%c") << "]";

                                if(!msg.sModule.empty())
                                    clipboard << " (" << msg.sModule << "):";

                                clipboard << "\t" << msg.sMessage
                                     << "\tThread: " << std::setbase(16) << msg.threadId << std::setbase(10);

                                if(copyFilename && !msg.sFilePosition.empty())
                                    clipboard << "\t@File: " << msg.sFilePosition;

                                clipboard << std::endl;
                            }
                            ImGui::SetClipboardText(clipboard.str().c_str());

                            ImGui::CloseCurrentPopup();
                        }
                        ImGui::PushItemFlag(ImGuiItemFlags_SelectableDontClosePopup, true);
                        ImGui::MenuItem("enable copy filename",nullptr,&copyFilename);
                        ImGui::PopItemFlag();
                        ImGui::EndPopup();
                    }

                    // draw lvl and module
                    ImGui::SameLine();
                    if(!msg.plaintext)
                    {
                        ImGui::TextColored(logLevelToColor(msg.lvl), "[%s]", toString(msg.lvl).c_str());
                        ImGui::NextColumn();

                        ImGui::Text("(%s)", msg.sModule.c_str());
                        ImGui::NextColumn();
                    } else
                    {
                        ImGui::NextColumn();
                        ImGui::NextColumn();
                    }

                    // draw actual text
                    ImGui::Text("%s", msg.sMessage.c_str());
                    ImGui::NextColumn();

                    ImGui::PopID();
                }

            // autoscroll, but set it for two frames
            if (autoscroll && buffer.filterChanged())
                ImGui::SetScrollHereY(1.0f);

        }
        ImGui::EndChild();
        ImGui::PopStyleVar();

    }

    if(drawAsChild)
        ImGui::EndChild();
    else
        ImGui::End();
}

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