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

    using Image16Resource = Resource<Image16>;
    using Image16RC = ResourceCache<Image16,Image16>;
    auto preloadImage16 = [](std::string data)
    {
        return make_unique<Image16>(reinterpret_cast<unsigned char*>(data.data()),data.size());
    };

    auto loadImage16 = [](std::unique_ptr<Image16> img) { return img;};


    ResourceManager< ImageRC,Image16RC > resourceManager( {preloadImage,loadImage,MPU_LIB_RESOURCE_PATH,
                                                 std::make_unique<Image8>(MPU_LIB_RESOURCE_PATH"missingTexture.png"),
                                                 "Image-8bit"},
                                                {preloadImage16,loadImage16,MPU_LIB_RESOURCE_PATH,
                                                 std::make_unique<Image16>(MPU_LIB_RESOURCE_PATH"missingTexture.png"),
                                                 "Image-16bit"}        );
    Image16Resource checker16 = resourceManager.load<Image16>("checker-map.png");
    checker16.unload();

    resourceManager.preload<Image8>("checker-map.png");
    resourceManager.preload<Image8>("../examples/resourceManagerTest/test_texture.png");
    resourceManager.preload<Image8>("checker-map_horizontal.png");
    resourceManager.preload<Image8>("checker-map_vertical.png");
    mpu::sleep_ms(500);
    mpu::SimpleStopwatch sw;
    ImageResource checker = resourceManager.load<Image8>("checker-map.png");
    ImageResource checker2 = resourceManager.load<Image8>("checker-map_horizontal.png");
    ImageResource checker3 = resourceManager.load<Image8>("checker-map_vertical.png");
    logINFO("Test") << "loading took " << sw.getSeconds();

    resourceManager.forceReload<Image8>("checker-map.png");

    gph::Sprite2D sprite(*checker);
    gph::Sprite2D sprite2(*checker2);

    sw.reset();
    ImageResource checker5 = resourceManager.load<Image8>("checker-map.png");
    ImageResource checker6 = resourceManager.load<Image8>("checker-map_horizontal.png");
    ImageResource checker7 = resourceManager.load<Image8>("checker-map_vertical.png");
    logINFO("Test") << "access took " << sw.getSeconds();

    gph::Sprite2D sprite4(*checker5);
    gph::Sprite2D sprite5(*checker6);

    checker7.unload();
    checker3.unload();

    ImageResource testTexture = resourceManager.load<Image8>("../examples/resourceManagerTest/test_texture.png");

    // Main loop
    while (window.frameEnd(), gph::Input::update(), window.frameBegin())
    {
        ImGui::ShowDemoWindow();
        cam.update();
        renderer.setView(cam.viewMatrix());

        mpu::gph::showResourceManagerDebugWindow(resourceManager);

        gph::Sprite2D sprite6(*testTexture);
        renderer.addSprite(sprite6);

        renderer.render();
    }

    return 0;
}