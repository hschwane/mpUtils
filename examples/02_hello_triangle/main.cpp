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

    // setup vertex array
    gph::VertexArray vao;
    // add a vertex array for our positions at location 0,
    // using the data from triangleVBO starting with the first element,
    // each vertex is of size sizeof(glm::vec4), and has 4 floating point vector components
    vao.addAttributeBufferArray( 0, 0, triangleVBO, 0, sizeof(glm::vec4), 4, 0);

    // setup the shader
    gph::ShaderProgram minimalShader({{PROJECT_SHADER_PATH"minimalVertexShader.vert"},{PROJECT_SHADER_PATH"minimalFragmentShader.frag"}});

    // we loop until the user wants the app to close indicated by window.frameBegin()
    // frameBegin also makes sure we are ready to draw things
    while (window.frameBegin())
    {
        // check if some input happened
        // eg a key was pressed or the mouse was moved
        gph::Input::update();

        // draw triangle
        minimalShader.use();
        vao.bind();
        glDrawArrays(GL_TRIANGLES,0,3); // draw 3 vertices, starting with the first

        // end the frame and display everything we have drawn to the screen
        window.frameEnd();
    }

    return 0;
}
