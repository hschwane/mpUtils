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
//    gph::enableVsync(true);

    glClearColor( .2f, .2f, .2f, 1.0f);

    gph::Renderer2D renderer;
    renderer.setProjection(-1,1,-1,1,0,10);

    gph::Sprite2D testSprite( MPU_LIB_RESOURCE_PATH"checker-map.png",glm::vec2(0.5f,0.5f),glm::radians(90.0f) );
    gph::Sprite2D testSprite2( MPU_LIB_RESOURCE_PATH"checker-map_vertical.png",glm::vec2(0.5f,0.25f),glm::radians(90.0f) );

    gph::Transform2D tf;
    bool lookAtCenter=false;

    // Main loop
    while (window.frameEnd(), gph::Input::update(), window.frameBegin())
    {
        renderer.addSprite( &testSprite, glm::mat4(1.0f),0);
        renderer.addRect( {0.5,1.0,0,1.0}, {1.0,1.0}, glm::scale(glm::vec3(1.0)), 3);

        ImGui::Begin("DebugWindow");
        ImGui::SliderFloat2("position", glm::value_ptr(tf.position), -1.0f,1.0f);
        ImGui::SliderFloat("orientation", &tf.orientation, -6.3f,6.3f);
        ImGui::Checkbox("look at center", &lookAtCenter);
        ImGui::Text("Angle to center: %f", gph::angleBetweenVectors2D(glm::vec2(1,0),glm::vec2(0,0)-tf.position));
        ImGui::End();

        if(lookAtCenter)
            tf.orientation = gph::lookAt2D(tf.position,glm::vec2(0,0));

        renderer.addSprite( &testSprite2, glm::mat4(tf),5);
        renderer.render();
    }

    return 0;
}