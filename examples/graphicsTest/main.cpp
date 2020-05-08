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
//    gph::enableVsync(true);

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

    gph::Sprite2D testSprite( MPU_LIB_RESOURCE_PATH"checker-map.png",false,glm::vec2(0.5f,0.5f),glm::vec2{0,0},glm::radians(0.0f) );
    gph::Sprite2D testSprite2( MPU_LIB_RESOURCE_PATH"checker-map_vertical.png",false,glm::vec2(0.25f,0.5f),glm::vec2{0.5,0},glm::radians(0.0f));

    gph::Transform2D tf;
    bool lookAtCenter=false;

    // Main loop
    while (window.frameEnd(), gph::Input::update(), window.frameBegin())
    {
        cam.showDebugWindow();
        cam.update();
        renderer.setView(cam.viewMatrix());

        renderer.addSprite( testSprite, glm::mat4(1.0f),0);
        renderer.addRect( {0.5,1.0,0,1.0}, {1.0,1.0}, glm::scale(glm::vec3(1.0)), 3);

        ImGui::Begin("DebugWindow");
        ImGui::SliderFloat2("position", glm::value_ptr(tf.position), -1.0f,1.0f);
        ImGui::SliderFloat("orientation", &tf.orientation, -6.3f,6.3f);
        ImGui::Checkbox("look at center", &lookAtCenter);
        ImGui::Text("Angle to center: %f", gph::angleBetweenVectors2D(glm::vec2(1,0),glm::vec2(0,0)-tf.position));
        ImGui::End();

        if(lookAtCenter)
            tf.orientation = gph::lookAt2D(tf.position,glm::vec2(0,0));

        renderer.addSprite( testSprite2, glm::mat4(tf),5);
        renderer.render();
    }

    return 0;
}