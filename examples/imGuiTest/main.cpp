//

//
//int main()
//{
//    Log myLog( LogLvl::ALL, ConsoleSink());
//    myLog.printHeader("imGuiTest", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");
//
//
//    int width = 800;
//    int height = 600;
//    gph::Window window(width, height,"mpUtils imGui test");
//
//    window.addSizeCallback([](int w, int h){ logDEBUG2("CallbackTest") << "window resized: " << w << "x" << h;});
//    window.addPositionCallback([](int x, int y){ logDEBUG2("CallbackTest") << "window position: " << x << ", " << y;});
//    window.addCloseCallback([](){ logDEBUG2("CallbackTest") << "window closed! ";});
//    window.addFocusCallback([](bool f){ if(f) { logDEBUG2("CallbackTest") << "window got focus!";} else {logDEBUG2("CallbackTest") << "window lost focus!";} });
//    window.addMinimizeCallback([](bool f){ if(f) { logDEBUG2("CallbackTest") << "window minimized!";} else {logDEBUG2("CallbackTest") << "window restored!";} });
//
//    bool p=false;
//    while(window.update())
//    {
//        if(window.getKey(GLFW_KEY_F11)==GLFW_PRESS && !p)
//        {
//            window.toggleFullscreen();
//            p = true;
//        } else if (window.getKey(GLFW_KEY_F11)==GLFW_RELEASE && p)
//            p=false;
//    }
//}

#include <mpUtils/mpUtils.h>
#include <mpUtils/mpGraphics.h>

using namespace mpu;

int main(int, char**)
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

    ImGui::create(window);
//    ImGui::StyleColorsDark();
//    ImGui::StyleCorporateGreyFlat();
    ImGui::StyleDarcula();
//    ImGui::StylePagghiu();
//    ImGui::StyleLightGreen();

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'misc/fonts/README.txt' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    auto io = ImGui::GetIO();
    io.Fonts->AddFontDefault();
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/Roboto-Medium.ttf", 16.0f);
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/Cousine-Regular.ttf", 15.0f);
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/DroidSans.ttf", 16.0f);
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/Karla-Regular.ttf", 16.0f);
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/ProggyTiny.ttf", 13.0f);

    bool show_demo_window = true;
    bool show_another_window = true;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Main loop
    while (window.update())
    {
        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);

    }

    return 0;
}
