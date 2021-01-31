/*
 * mpUtils
 * ImGuiWindows.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_IMGUIWINDOWS_H
#define MPUTILS_IMGUIWINDOWS_H

// includes
//--------------------
#include "ImGui.h"
#include "ImGuiElements.h"
#include "mpUtils/ResourceManager/ResourceManager.h"
#include "mpUtils/Log/BufferedSink.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

/**
 * @brief create and pass to the logger as a sink
 */
void showLoggerWindow(LogBuffer& buffer, bool* show= nullptr, bool drawAsChild=false);

/**
 * @brief window allows to switch between shipped styled and open the imgui style editor
 */
void showStyleSelectorWindow(bool* show= nullptr, bool drawAsChild=false);

/**
 * @brief simple overlay that shows fps in top left corner
 * @brief distance distance from selected corner
 * @brief corner 0= top left, 1= top right, 2 = bottom left, 3 = bottom right
 */
void showFPSOverlay(int corner = 0, ImVec2 distance={5.0f,5.0f});

/**
 * @brief window to display fps and ms per frame
 */
void showBasicPerformanceWindow(bool* show= nullptr, bool drawAsChild=false);

/**
 * @brief debug window to monitor and manage the resource manager, do not open multiple windows
 */
template <typename ... CacheTs>
void showResourceManagerDebugWindow( ResourceManager<CacheTs...>& resourceManager,
                                        bool* show=nullptr, bool drawAsChild=false);

// template function definition
//-------------------------------------------------------------------

template <typename... CacheTs>
void showResourceManagerDebugWindow( ResourceManager<CacheTs...>& resourceManager,
                                     bool* show, bool drawAsChild)
{
    using HandleType = typename ResourceManager<CacheTs...>::HandleType;

    bool visible;
    if(drawAsChild)
    {
        visible = ImGui::BeginChild("Resource Manager Debug Information");
    }
    else
    {
        ImGui::SetNextWindowSize(ImVec2(460,250),ImGuiCond_FirstUseEver);
        visible = ImGui::Begin("Resource Manager Debug Information", show);
    }

    // draw window content if visible
    if(visible)
    {
        static constexpr char const * stateNames[] = {"none","preloading","preloaded","preloadFailed","loading",
                                                      "failed","ready","defaulted"};

        // setup split
        float wndWidth = ImGui::GetWindowContentRegionWidth();
        static float oldWndWidth = wndWidth;
        static float widthLeft = wndWidth*0.33f;
        static float widthRight = wndWidth - widthLeft - ImGui::GetStyle().ItemSpacing.x;
        if(wndWidth != oldWndWidth)
        {
            widthLeft = wndWidth * (widthLeft / oldWndWidth);
            widthRight = wndWidth - widthLeft - ImGui::GetStyle().ItemSpacing.x;
            oldWndWidth = wndWidth;
        }
        ImGui::Splitter(true,2,&widthLeft,&widthRight,60,60,-1,false);

        // keep track of selected item
        static HandleType selectedHandle = 0;
        static std::string selectedCache;
        static std::string selectedName;
        // work around gcc bug
        HandleType thisSelectedHandle = selectedHandle;
        std::string thisSelectedCache = selectedCache;
        std::string thisSelectedName = selectedName;

        // on the left we draw a filterable list of all resources
        ImGui::BeginChild("left",ImVec2(widthLeft,0));
        {
            // header with buttons and filter
            bool collapseAll = false;
            bool expandAll = false;
            expandAll = ImGui::Button(ICON_FA_PLUS_SQUARE_O);
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Expand all");
            ImGui::SameLine();

            collapseAll = ImGui::Button(ICON_FA_MINUS_SQUARE_O);
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Collapse all");

            ImGui::SameLine();
            static ImGuiTextFilter filter;
            filter.Draw("##filter",ImGui::GetContentRegionAvail().x);
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("filter (include,-exclude)");

            auto drawSublist = [&](auto& resourceCache)
            {
                if(filter.IsActive() || expandAll)
                    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
                else if(collapseAll)
                    ImGui::SetNextItemOpen(false, ImGuiCond_Always);
                else
                    ImGui::SetNextItemOpen(true, ImGuiCond_Once);

                bool couldBeSelected = (thisSelectedCache == resourceCache.getDebugName());
                bool drawSubtree = ImGui::TreeNode( (resourceCache.getDebugName() + " - " + std::to_string(resourceCache.numLoaded())) .c_str() );
                if(ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::Text("Name: %s", resourceCache.getDebugName().c_str());
                    ImGui::Text("Loaded Elements: %i", resourceCache.numLoaded());
                    ImGui::Text("Working dir: %s", resourceCache.getWorkDir().c_str());
                    ImGui::EndTooltip();
                }

                if(drawSubtree)
                {
                    resourceCache.doForEachResource([&](const std::string& path, auto handle)
                    {
                        bool selected = (couldBeSelected && handle == thisSelectedHandle);
                        if(filter.PassFilter(path.c_str()) || selected)
                        {
                            if(ImGui::Selectable(path.c_str(), selected))
                            {
                                thisSelectedHandle = handle;
                                thisSelectedName = path;
                                thisSelectedCache = resourceCache.getDebugName();
                            }
                            if(ImGui::IsItemHovered())
                                ImGui::SetTooltip("%s",path.c_str());
                        }
                    });
                    ImGui::TreePop();
                }
            };

            ImGui::BeginChild("resource list",ImVec2(0,-ImGui::GetTextLineHeightWithSpacing()),true);
            {
                // execute draw sublist for every resource type
                int t[] = {0, ((void)(drawSublist(resourceManager.template get<typename CacheTs::ResourceType>())), 1)...};
                (void)t[0]; // silence compiler warning about t being unused
            }
            ImGui::EndChild();

            ImGui::BeginHorizontal("horizontal",ImGui::GetContentRegionAvail());
            ImGui::Spring();
            ImGui::Text("%i resources loaded", resourceManager.numLoaded());
            ImGui::Spring();
            ImGui::EndHorizontal();
        }
        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::BeginChild("right side", ImVec2(widthRight,0));
        {
            // draw some general buttons
            ImGui::AlignTextToFramePadding();
            ImGui::Text("threads:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(30);

            int numThreads = resourceManager.getNumThreads();
            if(ImGui::InputInt("|##threadshiddenlabel",&numThreads,0,0))
                resourceManager.setNumThreads(numThreads) ;

            ImGui::SameLine();
            if(ImGui::Button( ICON_FA_REFRESH " all"))
                resourceManager.forceReloadAll();
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("reload all");

            ImGui::SameLine();
            if(ImGui::Button( ICON_FA_TRASH_O " all"))
                resourceManager.tryReleaseAll();
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("remove all unused");

            auto drawSelectedResource = [&](auto& resourceCache)
            {
                if(thisSelectedCache == resourceCache.getDebugName())
                {
                    auto resourceInfo = resourceCache.getResourceInfo(thisSelectedHandle);

                    if(ImGui::Button(ICON_FA_REFRESH))
                        resourceCache.forceReload(thisSelectedName);
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("reload this resource");

                    ImGui::SameLine();
                    if(std::get<2>(resourceInfo) != 0)
                        ImGui::pushDisabled();
                    if(ImGui::Button(ICON_FA_TRASH_O))
                    {
                        resourceCache.tryRelease(thisSelectedName);
                        thisSelectedName ="";
                        thisSelectedCache ="";
                        thisSelectedHandle = 0;
                    }
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("remove this resource");
                    if(std::get<2>(resourceInfo) != 0)
                        ImGui::popDisabled();

                    ImGui::SameLine();
                    ImGui::Text("%s", thisSelectedName.c_str());

                    ImGui::Columns(2);
                    ImGui::Separator();

                    ImGui::Text("Name:");
                    ImGui::NextColumn();
                    ImGui::Text("%s", thisSelectedName.c_str());
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", thisSelectedName.c_str());
                    ImGui::NextColumn();
                    ImGui::Separator();

                    ImGui::Text("Type:");
                    ImGui::NextColumn();
                    ImGui::Text("%s", resourceCache.getDebugName().c_str());
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", resourceCache.getDebugName().c_str());
                    ImGui::NextColumn();
                    ImGui::Separator();

                    ImGui::Text("Path:");
                    ImGui::NextColumn();
                    ImGui::Text("%s", (resourceCache.getWorkDir() + thisSelectedName).c_str());
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", (resourceCache.getWorkDir() + thisSelectedName).c_str());
                    ImGui::NextColumn();
                    ImGui::Separator();

                    ImGui::Text("State:");
                    ImGui::NextColumn();
                    ImGui::Text("%s", stateNames[static_cast<int>(std::get<3>(resourceInfo))]);
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", stateNames[static_cast<int>(std::get<3>(resourceInfo))]);
                    ImGui::NextColumn();
                    ImGui::Separator();

                    ImGui::Text("References:");
                    ImGui::NextColumn();
                    ImGui::Text("%i", std::get<2>(resourceInfo));
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("%i", std::get<2>(resourceInfo));
                    ImGui::NextColumn();
                    ImGui::Separator();

                    ImGui::Text("Adress");
                    ImGui::NextColumn();
                    ImGui::Text("%p", std::get<0>(resourceInfo));
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("%p", std::get<0>(resourceInfo));
                    ImGui::NextColumn();
                    ImGui::Separator();

                    ImGui::Text("Preload Adress");
                    ImGui::NextColumn();
                    ImGui::Text("%p", std::get<1>(resourceInfo));
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("%p", std::get<1>(resourceInfo));
                    ImGui::Separator();

                    ImGui::Columns(1);
                }
            };

            // now show info on the selcted resource
            ImGui::BeginChild("selected resource", ImVec2(0, 0), true);
            {
                // try to draw selected resource for all resource managers
                int t[] = {0, ((void)(drawSelectedResource(resourceManager.template get<typename CacheTs::ResourceType>())), 1)...};
                (void)t[0]; // silence compiler warning about t being unused
            }
            ImGui::EndChild();

        }
        ImGui::EndChild();

        // work around gcc bug
        selectedHandle = thisSelectedHandle;
        selectedCache = thisSelectedCache;
        selectedName = thisSelectedName;
    }

    if(drawAsChild)
        ImGui::EndChild();
    else
        ImGui::End();
}

}}
#endif //MPUTILS_IMGUIWINDOWS_H
