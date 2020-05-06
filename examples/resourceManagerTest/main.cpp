/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utilities
 * A utility library for my personal c++ projects
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

/*
 * This project is used by the developers to play with and test new features
 */

#include <mpUtils/mpUtils.h>
#include <mpUtils/mpGraphics.h>

using namespace mpu;
using namespace std;
using namespace std::chrono;

int main()
{
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("graphicsTest", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");

    int width = 800;
    int height = 800;
    gph::Window window(width, height,"mpUtils imGui test");

    ImGui::create(window);
    gph::enableVsync(true);

    glClearColor( .2f, .2f, .2f, 1.0f);

    gph::Camera2D cam;
    cam.addInputs();
    gph::Input::mapScrollToInput("Camera2DZoom");
    gph::Input::mapKeyToInput("Camera2DMoveDownUp",GLFW_KEY_W,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::positive);
    gph::Input::mapKeyToInput("Camera2DMoveDownUp",GLFW_KEY_S,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::negative);
    gph::Input::mapKeyToInput("Camera2DMoveLeftRight",GLFW_KEY_D,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::positive);
    gph::Input::mapKeyToInput("Camera2DMoveLeftRight",GLFW_KEY_A,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::negative);

    gph::Renderer2D renderer;
    renderer.setSamplingLinear(false,false);
    window.addFBSizeCallback([&](int w, int h)
    {
        glViewport(0,0,w,h);
        float aspect = float(w)/h;
        renderer.setProjection(-1*aspect,1*aspect,-1,1);
    });
    float aspect = float(width)/height;
    renderer.setProjection(-1*aspect,1*aspect,-1,1);

    // test resource management
    using ImageResource = Resource<Image8>;
    using ImageRC = ResourceCache<Image8,Image8>;
    auto preloadImage = [](std::string data)
            {
                return make_unique<Image8>(reinterpret_cast<unsigned char*>(data.data()),data.size());
            };

    auto loadImage = [](std::unique_ptr<Image8> img) { return img;};

    mpu::ThreadPool p(2);
    auto addTask = [&p](const std::function<void()>& f){p.enqueue(f);};

    ImageRC imageCache(preloadImage,loadImage,MPU_LIB_RESOURCE_PATH,addTask,std::make_unique<Image8>(MPU_LIB_RESOURCE_PATH"missingTexture.png"));

    imageCache.preload("checker-map.png");
    imageCache.preload("../examples/resourceManagerTest/test_texture.png");
    imageCache.preload("checker-map_horizontal.png");
    imageCache.preload("checker-map_vertical.png");
    mpu::sleep_ms(500);
    mpu::SimpleStopwatch sw;
    ImageResource checker = imageCache.load("checker-map.png");
    ImageResource checker2 = imageCache.load("checker-map_horizontal.png");
    ImageResource checker3 = imageCache.load("checker-map_vertical.png");
    logINFO("Test") << "loading took " << sw.getSeconds();

    imageCache.forceReload("checker-map.png");

    gph::Sprite2D sprite(*checker);
    gph::Sprite2D sprite2(*checker2);

    sw.reset();
    ImageResource checker5 = imageCache.load("checker-map.png");
    ImageResource checker6 = imageCache.load("checker-map_horizontal.png");
    ImageResource checker7 = imageCache.load("checker-map_vertical.png");
    logINFO("Test") << "access took " << sw.getSeconds();

    gph::Sprite2D sprite4(*checker5);
    gph::Sprite2D sprite5(*checker6);

    checker7.unload();
    checker3.unload();

    ImageResource testTexture = imageCache.load("../examples/resourceManagerTest/test_texture.png");

    // Main loop
    while (window.frameEnd(), gph::Input::update(), window.frameBegin())
    {
        ImGui::ShowDemoWindow();
        cam.update();
        renderer.setView(cam.viewMatrix());

        if(ImGui::Begin("Resources"))
        {
            static constexpr char const * stateNames[] = {"none","preloading","preloaded","preloadFailed","loading","failed","ready","defaulted"};
            static ImageRC::HandleType selected = 0;
            static std::string selectedPath;
            ImGui::BeginChild("resource list",ImVec2(240,0), true);
            {
                static ImGuiTextFilter filter;
                filter.Draw();

                imageCache.doForEachResource([](const std::string& path, ImageRC::HandleType handle)
                                             {
                                                    bool isSelected = selected == handle;
                                                 if(filter.PassFilter(path.c_str()) || isSelected)
                                                     if(ImGui::Selectable(path.c_str(), isSelected) || isSelected)
                                                     {
                                                         selected = handle;
                                                         selectedPath = path;
                                                     }
                                             });
            }
            ImGui::EndChild();

            ImGui::SameLine();
            ImGui::BeginGroup();
            {
                auto resourceInfo = imageCache.getResourceInfo(selected);

                static int numThreads = 2;
                ImGui::AlignTextToFramePadding();
                ImGui::Text("threads:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90);
                if(ImGui::InputInt("|##threadshiddenlabel",&numThreads,1,2))
                    ; //TODO: change thread count;

                ImGui::SameLine();
                if(ImGui::Button("Reload all"))
                    imageCache.forceReloadAll();

                ImGui::SameLine();
                if(ImGui::Button("Release all unused"))
                    imageCache.tryReleaseAll();

                ImGui::BeginChild("selected resource", ImVec2(0, 0), true);

                if(ImGui::Button("Force reload"))
                    imageCache.forceReload(selectedPath);

                ImGui::SameLine();
                if(std::get<2>(resourceInfo) != 0)
                    ImGui::pushDisabled();
                if(ImGui::Button("Release"))
                    imageCache.tryRelease(selectedPath);
                if(std::get<2>(resourceInfo) != 0)
                    ImGui::popDisabled();

                ImGui::Columns(2);
                ImGui::Separator();

                ImGui::Text("Path:");
                ImGui::NextColumn();
                ImGui::Text("%s", selectedPath.c_str());
                ImGui::NextColumn();
                ImGui::Separator();

                ImGui::Text("State:");
                ImGui::NextColumn();
                ImGui::Text("%s", stateNames[static_cast<int>(std::get<3>(resourceInfo))]);
                ImGui::NextColumn();
                ImGui::Separator();

                ImGui::Text("References:");
                ImGui::NextColumn();
                ImGui::Text("%i", std::get<2>(resourceInfo));
                ImGui::NextColumn();
                ImGui::Separator();

                ImGui::Text("Adress");
                ImGui::NextColumn();
                ImGui::Text("%p", std::get<0>(resourceInfo));
                ImGui::NextColumn();
                ImGui::Separator();

                ImGui::Text("Preload Adress");
                ImGui::NextColumn();
                ImGui::Text("%p", std::get<1>(resourceInfo));
                ImGui::Separator();

                ImGui::Columns(1);
                ImGui::EndChild();
            }
            ImGui::EndGroup();
        }
        ImGui::End();

        gph::Sprite2D sprite6(*testTexture);
        renderer.addSprite(sprite6);

        renderer.render();
    }

    return 0;
}