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
//    renderer.setProjection(-1,1,-1,1,0,10);

    gph::SpriteInstance2D testSprite( std::make_shared<gph::Sprite2D>(MPU_LIB_RESOURCE_PATH"checker-map.png",glm::vec2(0.5f,0.5f),glm::radians(90.0f)) );
    gph::SpriteInstance2D testSprite2( std::make_shared<gph::Sprite2D>(MPU_LIB_RESOURCE_PATH"checker-map_vertical.png",glm::vec2(0.5f,0.25f),glm::radians(90.0f)) );

    gph::Transform2D tf;
    bool lookAtCenter=false;

    gph::Camera cam(gph::Camera::fps);
    cam.addInputs();


    gph::Input::mapKeyToInput("CameraMoveSideways",GLFW_KEY_D,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::positive);
    gph::Input::mapKeyToInput("CameraMoveSideways",GLFW_KEY_A,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::negative);
    gph::Input::mapKeyToInput("CameraMoveForwardBackward",GLFW_KEY_W,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::positive);
    gph::Input::mapKeyToInput("CameraMoveForwardBackward",GLFW_KEY_S,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::negative);
    gph::Input::mapKeyToInput("CameraMoveUpDown",GLFW_KEY_Q,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::negative);
    gph::Input::mapKeyToInput("CameraMoveUpDown",GLFW_KEY_E,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::positive);

    gph::Input::mapCourserToInput("CameraPanHorizontal", gph::Input::AxisOrientation::horizontal,gph::Input::AxisBehavior::negative,0, "EnablePan");
    gph::Input::mapCourserToInput("CameraPanVertical", gph::Input::AxisOrientation::vertical,gph::Input::AxisBehavior::positive,0, "EnablePan");
    gph::Input::mapScrollToInput("CameraZoom");

    gph::Input::mapMouseButtonToInput("EnablePan", GLFW_MOUSE_BUTTON_MIDDLE);
    gph::Input::mapKeyToInput("EnablePan", GLFW_KEY_LEFT_ALT);

    gph::Input::mapCourserToInput("CameraRotateHorizontal", gph::Input::AxisOrientation::horizontal,gph::Input::AxisBehavior::negative,0, "EnableRotation");
    gph::Input::mapCourserToInput("CameraRotateVertical", gph::Input::AxisOrientation::vertical,gph::Input::AxisBehavior::negative,0, "EnableRotation");

    gph::Input::mapMouseButtonToInput("EnableRotation", GLFW_MOUSE_BUTTON_LEFT);
    gph::Input::mapKeyToInput("EnableRotation", GLFW_KEY_LEFT_CONTROL);

    gph::Input::mapKeyToInput("CameraMovementSpeed",GLFW_KEY_RIGHT_BRACKET,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::positive);
    gph::Input::mapKeyToInput("CameraMovementSpeed",GLFW_KEY_SLASH,gph::Input::ButtonBehavior::whenDown,gph::Input::AxisBehavior::negative);
    gph::Input::mapKeyToInput("CameraToggleMode",GLFW_KEY_R);
    gph::Input::mapKeyToInput("CameraSlowMode",GLFW_KEY_LEFT_SHIFT,gph::Input::ButtonBehavior::whenDown);
    gph::Input::mapKeyToInput("CameraFastMode",GLFW_KEY_SPACE,gph::Input::ButtonBehavior::whenDown);

    // Main loop
    while (window.frameEnd(), gph::Input::update(), window.frameBegin())
    {
        cam.showDebugWindow();
        cam.update();
        renderer.setProjection( glm::perspective(glm::radians(90.0f),1.0f,0.1f,100.0f) * cam.viewMatrix() );

        renderer.addSprite( testSprite,glm::mat4(1.0f),2);

        ImGui::Begin("DebugWindow");
        ImGui::SliderFloat2("position", glm::value_ptr(tf.position), -1.0f,1.0f);
        ImGui::SliderFloat("orientation", &tf.orientation, -6.3f,6.3f);
        ImGui::Checkbox("look at center", &lookAtCenter);
        ImGui::Text("Angle to center: %f", gph::angleBetweenVectors2D(glm::vec2(1,0),glm::vec2(0,0)-tf.position));
        ImGui::End();

        if(lookAtCenter)
            tf.orientation = gph::lookAt2D(tf.position,glm::vec2(0,0));

        renderer.addSprite( testSprite2, glm::mat4(tf));
        renderer.render();
    }

    return 0;
}