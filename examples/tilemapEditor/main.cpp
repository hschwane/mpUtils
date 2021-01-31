#include "TilemapEditor.h"

int main(int, char**)
{
    using namespace mpu;

    LogBuffer buffer(1000);
    Log myLog( LogLvl::ALL, ConsoleSink(), BufferedSink(buffer));
    myLog.printHeader("tilemapEditor", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");

    gph::Window wnd(10, 10,"mpUtils tilemap editor");
    ImGui::create(wnd);
    gph::enableVsync(true);

    fs::path startDir = fs::current_path();
    startDir += "/imgui.ini";
    ImGui::GetIO().IniFilename = startDir.c_str();

    // setup renderer
    mpu::gph::Renderer2D renderer;
    renderer.setSamplingLinear(true,false);

    // handle window resizing
    glm::ivec2 wndSize;
    wnd.addFBSizeCallback([&](int w, int h)
        {
            wndSize = {w,h};
            glViewport(0,0,w,h);
            float aspect = float(w)/h;
            renderer.setProjection(-1*aspect,1*aspect,-1,1);
        });
    wnd.setSize(800,800); // trigger resize callback to set projection

    // set grey background for startup
    glClearColor(0.2,0.2,0.2,1.0);

    TilemapEditor editor(buffer);
//    fs::path workDir = MAP_WORKDIR;
//    auto noneTile = std::make_shared<gph::TileType>("none", gph::getEmptySprite());
//    gph::Tilemap activeMap({0,0},{1.0,1.0},{30,30},gph::AutotileMode::none, noneTile);

    // Main loop
    while(wnd.frameEnd(), gph::Input::update(), wnd.frameBegin()) {
        editor.run(renderer);
    }
    return 0;
}
