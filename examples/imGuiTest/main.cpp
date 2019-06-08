#include <mpUtils/mpUtils.h>
#include <mpUtils/mpGraphics.h>

using namespace mpu;

int main()
{
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("imGuiTest", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");


    int width = 800;
    int height = 600;
    gph::Window window(width, height,"mpUtils imGui test");

    window.addSizeCallback([](int w, int h){ logDEBUG2("CallbackTest") << "window resized: " << w << "x" << h;});
    window.addPositionCallback([](int x, int y){ logDEBUG2("CallbackTest") << "window position: " << x << ", " << y;});
    window.addCloseCallback([](){ logDEBUG2("CallbackTest") << "window closed! ";});
    window.addFocusCallback([](bool f){ if(f) { logDEBUG2("CallbackTest") << "window got focus!";} else {logDEBUG2("CallbackTest") << "window lost focus!";} });
    window.addMinimizeCallback([](bool f){ if(f) { logDEBUG2("CallbackTest") << "window minimized!";} else {logDEBUG2("CallbackTest") << "window restored!";} });

    bool p=false;
    while(window.update())
    {
        if(window.getKey(GLFW_KEY_F11)==GLFW_PRESS && !p)
        {
            window.toggleFullscreen();
            p = true;
        } else if (window.getKey(GLFW_KEY_F11)==GLFW_RELEASE && p)
            p=false;
    }
}