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
    ImageRC* imageRc = nullptr;
    ResourceManager< ImageRC,Image16RC,gph::Sprite2DRC > resourceManager( {preloadImage,finalLoadImage,MPU_LIB_RESOURCE_PATH,
                                                            getDefaultImage(), "Image-8bit"},
                                                {preloadImage16,finalLoadImage16,MPU_LIB_RESOURCE_PATH,
                                                 getDefaultImage16(), "Image-16bit"},
                                                 { [&](const std::string& data){ return gph::preloadSprite(imageRc,data); },
                                                  gph::finalLoadSprite2D,
                                                  MPU_LIB_RESOURCE_PATH, gph::getDefaultSprite(), "Sprite2D"} );
    imageRc = &resourceManager.get<Image8>();

    std::shared_ptr<Image16> checker16 = resourceManager.load<Image16>("checker-map.png");
    checker16 = nullptr;

    resourceManager.preload<Image8>("checker-map.png");
    resourceManager.preload<Image8>("../examples/resourceManagerTest/test_texture.png");
    resourceManager.preload<Image8>("checker-map_horizontal.png");
    resourceManager.preload<Image8>("checker-map_vertical.png");
    mpu::sleep_ms(500);
    mpu::SimpleStopwatch sw;
    std::shared_ptr<Image8> checker = resourceManager.load<Image8>("checker-map.png");
    std::shared_ptr<Image8> checker2 = resourceManager.load<Image8>("checker-map_horizontal.png");
    std::shared_ptr<Image8> checker3 = resourceManager.load<Image8>("checker-map_vertical.png");
    logINFO("Test") << "loading took " << sw.getSeconds();

    resourceManager.forceReload<Image8>("checker-map.png");

    gph::Sprite2D sprite(*checker);
    gph::Sprite2D sprite2(*checker2);

    sw.reset();
    std::shared_ptr<Image8> checker5 = resourceManager.load<Image8>("checker-map.png");
    std::shared_ptr<Image8> checker6 = resourceManager.load<Image8>("checker-map_horizontal.png");
    std::shared_ptr<Image8> checker7 = resourceManager.load<Image8>("checker-map_vertical.png");
    logINFO("Test") << "access took " << sw.getSeconds();

    gph::Sprite2D sprite4(*checker5);
    gph::Sprite2D sprite5(*checker6);

    checker7 = nullptr;
    checker3 = nullptr;

    std::shared_ptr<Image8> testTexture = resourceManager.load<Image8>("../examples/resourceManagerTest/test_texture.png");

    // try to load some sprites
    std::shared_ptr<gph::Sprite2D> spriteReource = resourceManager.load<gph::Sprite2D>("../examples/resourceManagerTest/testSprite.sprite");

    gph::Sprite2D sprite6(*testTexture);

    // Main loop
    while (window.frameEnd(), gph::Input::update(), window.frameBegin())
    {
        ImGui::ShowDemoWindow();
        cam.update();
        renderer.setView(cam.viewMatrix());

        mpu::gph::showBasicPerformanceWindow();
        mpu::gph::showResourceManagerDebugWindow(resourceManager);

        renderer.addSprite(*spriteReource,glm::translate(glm::vec3(0.5f,0.5f,0.0f)));

        renderer.addSprite(sprite6);

        renderer.render();
    }

    return 0;
}