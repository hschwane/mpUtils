/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 *
 * Copyright 2020 Hendrik Schwanekamp
 *
 */

/*
 * Example on setting up a simple graphics project
 */

#include <mpUtils/mpUtils.h>
#include <mpUtils/mpGraphics.h>

// binds keys to camera functions
void setKeybindings()
{
    using namespace mpu::gph;

    // camera
    Input::mapKeyToInput("CameraMoveSideways",GLFW_KEY_D,Input::ButtonBehavior::whenDown,Input::AxisBehavior::positive);
    Input::mapKeyToInput("CameraMoveSideways",GLFW_KEY_A,Input::ButtonBehavior::whenDown,Input::AxisBehavior::negative);
    Input::mapKeyToInput("CameraMoveForwardBackward",GLFW_KEY_W,Input::ButtonBehavior::whenDown,Input::AxisBehavior::positive);
    Input::mapKeyToInput("CameraMoveForwardBackward",GLFW_KEY_S,Input::ButtonBehavior::whenDown,Input::AxisBehavior::negative);
    Input::mapKeyToInput("CameraMoveUpDown",GLFW_KEY_Q,Input::ButtonBehavior::whenDown,Input::AxisBehavior::negative);
    Input::mapKeyToInput("CameraMoveUpDown",GLFW_KEY_E,Input::ButtonBehavior::whenDown,Input::AxisBehavior::positive);

    Input::mapCourserToInput("CameraPanHorizontal", Input::AxisOrientation::horizontal,Input::AxisBehavior::negative,0, "EnablePan");
    Input::mapCourserToInput("CameraPanVertical", Input::AxisOrientation::vertical,Input::AxisBehavior::positive,0, "EnablePan");
    Input::mapScrollToInput("CameraZoom");

    Input::mapMouseButtonToInput("EnablePan", GLFW_MOUSE_BUTTON_MIDDLE);
    Input::mapKeyToInput("EnablePan", GLFW_KEY_LEFT_ALT);

    Input::mapCourserToInput("CameraRotateHorizontal", Input::AxisOrientation::horizontal,Input::AxisBehavior::negative,0, "EnableRotation");
    Input::mapCourserToInput("CameraRotateVertical", Input::AxisOrientation::vertical,Input::AxisBehavior::negative,0, "EnableRotation");

    Input::mapMouseButtonToInput("EnableRotation", GLFW_MOUSE_BUTTON_LEFT);
    Input::mapKeyToInput("EnableRotation", GLFW_KEY_LEFT_CONTROL);

    Input::mapKeyToInput("CameraMovementSpeed",GLFW_KEY_RIGHT_BRACKET,Input::ButtonBehavior::whenDown,Input::AxisBehavior::positive);
    Input::mapKeyToInput("CameraMovementSpeed",GLFW_KEY_SLASH,Input::ButtonBehavior::whenDown,Input::AxisBehavior::negative);
    Input::mapKeyToInput("CameraToggleMode",GLFW_KEY_R);
    Input::mapKeyToInput("CameraSlowMode",GLFW_KEY_LEFT_SHIFT,Input::ButtonBehavior::whenDown);
    Input::mapKeyToInput("CameraFastMode",GLFW_KEY_SPACE,Input::ButtonBehavior::whenDown);
}

int main()
{
    using namespace mpu;

    // initialize logging, errors and other messages will be written to this log
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("Example1", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");

    // set the openGL version needed for the features we use
    gph::Window::setGlVersion(4,5);

    // create a window
    gph::Window window(800, 800,"OpenGL Example");

    // set background color
    glClearColor( .2f, .2f, .2f, 1.0f);

    // limit framerate to monitor framerate
    gph::enableVsync(true);

    // setup the triangle
    glm::vec4 triangle[] =  {{-0.5,-0.5,0,1},
                             {0.5,-0.5,0,1},
                             {0.0,0.5,0,1}};
    gph::Buffer<glm::vec4> triangleVBO(triangle,3);
    // add additional values to the vertices
    float density[] = {0.1,4,10};
    gph::Buffer<float> densityVBO(density,3);

    // setup vertex array
    gph::VertexArray vao;
    // add a vertex array for our positions at location 0,
    // using the data from triangleVBO starting with the first element,
    // each vertex is of size sizeof(glm::vec4), and has 4 floating point vector components
    vao.addAttributeBufferArray( 0, 0, triangleVBO, 0, sizeof(glm::vec4), 4);
    // add vertex array for density
    vao.addAttributeBufferArray( 1, 1, densityVBO, 0, sizeof(float), 1);

    // setup the shader
    gph::ShaderProgram shader({{PROJECT_SHADER_PATH"minimal3DVS.vert"},{PROJECT_SHADER_PATH"minimalColorFS.frag"}});
    float maxDensity = 10;
    shader.uniform1f("maxDensity",maxDensity);

    // object transformation
    glm::vec3 triPosition(0,0,-1);
    glm::vec3 triRotation(0,0,0);
    glm::vec3 triPosition2(2,2,-5);
    glm::vec3 triRotation2(0,0,0);

    // perspective projection matrix
    glm::mat4 projectionMat = glm::perspective(glm::radians(60.0f),1.0f,0.01f,10.0f);
    shader.uniformMat4("projectionMat",projectionMat);

    // setup camera
    gph::Camera cam;
    cam.addInputs();
    setKeybindings();

    // add resize callback to respond to window resizing
    window.addSizeCallback([&](int w, int h)
    {
        // set the viewport
        glViewport(0,0,w,h);
        // update perspective projection matrix
        glm::mat4 projectionMat = glm::perspective(glm::radians(60.0f),float(w)/h,0.01f,10.0f);
        shader.uniformMat4("projectionMat",projectionMat);
    });

    // initialize user interface
    ImGui::create(window);

    // enable depth test
    // without that the last drawn object will always be on top of the others
    glEnable(GL_DEPTH_TEST);

    // we loop until the user wants the app to close indicated by window.frameBegin()
    // frameBegin also makes sure we are ready to draw things
    while (window.frameBegin())
    {
        // check if some input happened
        // eg a key was pressed or the mouse was moved
        gph::Input::update();

        // user interface
        ImGui::Begin("Demo");
            if(ImGui::DragFloat("maxDensity",&maxDensity,0.01))
            {
                shader.uniform1f("maxDensity",maxDensity);
            }
            ImGui::DragFloat3("position",glm::value_ptr(triPosition),0.01);
            ImGui::DragFloat3("rotation",glm::value_ptr(triRotation),0.1);
            ImGui::DragFloat3("position2",glm::value_ptr(triPosition2),0.01);
            ImGui::DragFloat3("rotation2",glm::value_ptr(triRotation2),0.1);
        ImGui::End();

        // set view matrix
        cam.update();
        shader.uniformMat4("viewMat",cam.viewMatrix());

        // build model matrix
        glm::mat4 modelMat = glm::eulerAngleXYZ(glm::radians(triRotation.x),glm::radians(triRotation.y),glm::radians(triRotation.z));
        modelMat = glm::translate(triPosition) * modelMat;
        shader.uniformMat4("modelMat",modelMat);

        // draw triangle
        shader.use();
        vao.bind();
        glDrawArrays(GL_TRIANGLES,0,3); // draw 3 vertices, starting with the first

        // draw another triangle at a different position
        // build model matrix
        modelMat = glm::eulerAngleXYZ(glm::radians(triRotation2.x),glm::radians(triRotation2.y),glm::radians(triRotation2.z));
        modelMat = glm::translate(triPosition2) * modelMat;
        shader.uniformMat4("modelMat",modelMat);
        glDrawArrays(GL_TRIANGLES,0,3); // draw 3 vertices, starting with the first

        // end the frame and display everything we have drawn to the screen
        window.frameEnd();
    }

    return 0;
}