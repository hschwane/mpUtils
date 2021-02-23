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

    // limit framerate to monitor framerate
    gph::enableVsync(true);

    // set background color
    glClearColor( .2f, .2f, .2f, 1.0f);

    // we loop until the user wants the app to close indicated by window.frameBegin()
    // frameBegin also makes sure we are ready to draw things
    while (window.frameBegin())
    {
        // check if some input happened
        // eg a key was pressed or the mouse was moved
        gph::Input::update();

        // end the frame and display everything we have drawn to the screen
        window.frameEnd();
    }

    return 0;
}