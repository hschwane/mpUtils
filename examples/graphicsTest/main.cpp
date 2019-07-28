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


    int width = 800;
    int height = 600;
    gph::Window window(width, height,"mpUtils imGui test");

    ImGui::create(window);
    gph::enableVsync(true);

    glClearColor( .2f, .2f, .2f, 1.0f);

    gph::ScreenFillingTri sft(MPU_LIB_SHADER_PATH"drawTexture.frag");
//    sft.shader().uniform4f("color", {1,0.5,0.5,1.0});
    auto tex = gph::makeTextureFromFile("/home/hendrik/Bilder/658142.jpg");
    tex->bind(GL_TEXTURE0);

    // Main loop
    while (window.frameEnd(), gph::Input::update(), window.frameBegin())
    {
        sft.draw();
    }

    return 0;
}