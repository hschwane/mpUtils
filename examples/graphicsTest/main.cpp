/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utillities
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


    int width = 600;
    int height = 600;
    gph::Window window(width, height,"mpUtils imGui test");

    ImGui::create(window);
    gph::enableVsync(true);

    glClearColor( .2f, .2f, .2f, 1.0f);

    gph::Renderer2D renderer;

    gph::SpriteInstance2D testSprite( std::make_shared<gph::Sprite2D>(MPU_LIB_RESOURCE_PATH"checker-map.png",glm::vec2(0.5f,0.5f),glm::radians(90.0f)) );

    // Main loop
    while (window.frameEnd(), gph::Input::update(), window.frameBegin())
    {
//        renderer.addSprite( glm::mat4(1.0f),testSprite);

        gph::Transform2D tf({-0.5f,-0.5f});
        tf.orientation = gph::lookAt2D(tf.position,glm::vec2(0,0));

        renderer.addSprite( glm::mat4(tf),testSprite);
        renderer.render();
    }

    return 0;
}