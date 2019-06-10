#include <mpUtils/mpUtils.h>
#include <mpUtils/mpGraphics.h>

using namespace mpu;

void setupInputs()
{
    gph::Input::addButton("TestButton","Tests the button", [](gph::Window& w)
    {
        logINFO("Test") << "Button triggered from window at " << glm::to_string(w.getPosition());
    }, gph::Input::ButtonBehavior::onPress);

    gph::Input::mapKeyToInput("TestButton", GLFW_KEY_A, 0, gph::Input::ButtonBehavior::onDoubleClick);
    gph::Input::mapKeyToInput("TestButton", GLFW_KEY_B, GLFW_MOD_ALT | GLFW_MOD_SHIFT);

    // there will be a warning since we swapped the order of calls, but it will work
    gph::Input::mapKeyToInput("SecondButton", GLFW_KEY_W);
    gph::Input::addButton("SecondButton","Tests the button", [](gph::Window& w)
    {
        logINFO("Test") << "Second button triggered from window at " << glm::to_string(w.getPosition());
    }, gph::Input::ButtonBehavior::onPress);


    gph::Input::addButton("mb","Tests the button", [](gph::Window& w)
    {
        if(w.getInputMode(GLFW_CURSOR) == GLFW_CURSOR_DISABLED)
            w.setInputMode( GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        else
            w.setInputMode( GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        logINFO("Test") << "Mouse button triggered from window at " << glm::to_string(w.getPosition());
    }, gph::Input::ButtonBehavior::onPress);
    gph::Input::mapMouseButtonToInput("mb", GLFW_MOUSE_BUTTON_RIGHT);


    gph::Input::addButton("TriggeredByScroll","A button that is triggered by scrolling", [](gph::Window& w)
    {
        logINFO("Test") << "Sroll Wheel was moved for window at: " << glm::to_string(w.getPosition());
    });
    gph::Input::mapScrollToInput("TriggeredByScroll",GLFW_MOD_CONTROL,gph::Input::AxisBehavior::positive);
    gph::Input::mapScrollToInput("TriggeredByScroll",GLFW_MOD_CONTROL,gph::Input::AxisBehavior::negative);


    gph::Input::addAxis("TestAxis","A axis that prints its value", [](gph::Window& w, double v)
    {
        logINFO("Test") << "Axis changed by " << v;
    });
    gph::Input::mapScrollToInput("TestAxis");

    gph::Input::addAxis("TestAxis2","A axis that prints its value", [](gph::Window& w, double v)
    {
        logINFO("Test") << "Axis changed by " << v;
    });
    gph::Input::mapCourserToInput("TestAxis2",gph::Input::AxisOrientation::horizontal);
}

int main(int, char**)
{
    Log myLog( LogLvl::ALL, ConsoleSink());
    myLog.printHeader("imGuiTest", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");

    int width = 800;
    int height = 600;
    gph::Window window(width, height,"mpUtils imGui test");

    window.addSizeCallback([](int w, int h){ logDEBUG2("CallbackTest") << "window resized: " << w << "x" << h;});
    window.addCloseCallback([](){ logDEBUG2("CallbackTest") << "window closed! ";});
    window.addFocusCallback([](bool f){ if(f) { logDEBUG2("CallbackTest") << "window got focus!";} else {logDEBUG2("CallbackTest") << "window lost focus!";} });
    window.addMinimizeCallback([](bool f){ if(f) { logDEBUG2("CallbackTest") << "window minimized!";} else {logDEBUG2("CallbackTest") << "window restored!";} });

    ImGui::create(window);
//    ImGui::StyleColorsDark();
//    ImGui::StyleCorporateGreyFlat();
    ImGui::StyleDarcula();
//    ImGui::StylePagghiu();
//    ImGui::StyleLightGreen();

    gph::enableVsync(true);

    // Load Fonts
    auto io = ImGui::GetIO();
    io.Fonts->AddFontDefault();
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/Roboto-Medium.ttf", 16.0f);
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/Cousine-Regular.ttf", 15.0f);
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/DroidSans.ttf", 16.0f);
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/Karla-Regular.ttf", 16.0f);
    io.Fonts->AddFontFromFileTTF(MPU_LIB_RESOURCE_PATH"/fonts/ProggyTiny.ttf", 13.0f);

    glClearColor( .2f, .2f, .2f, 1.0f);
    bool show_demo_window = false;
    std::unique_ptr<gph::Window> secondWindow;

    setupInputs();

    // Main loop
    while (window.update())
    {

        ImGui::Begin("Input Test");
        {

            ImGui::Text("CursorScreenPos: %s", glm::to_string(gph::Input::getCursorScreenPos()).c_str());
            ImGui::Text("Is \"0\" key pressed in any window?: %i", gph::Input::isKeyDown(GLFW_KEY_0));
            ImGui::Text("Is \"1\" key pressed in main window?: %i", window.isKeyDown(GLFW_KEY_1));
            ImGui::Text("Is left mouse button pressed in any window?: %i", gph::Input::isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT));
            ImGui::Text("Is left mouse button pressed in main window?: %i", window.isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT));

            if(secondWindow)
                ImGui::Text("Is \"2\" key pressed in second window?: %i", secondWindow->isKeyDown(GLFW_KEY_2));

            ImGui::Checkbox("Show Demo Window", &show_demo_window);

            static int cursorId = GLFW_ARROW_CURSOR;
            if(ImGui::BeginCombo("", "Select Cursor"))
            {
                if(ImGui::Selectable("ARROW", cursorId == GLFW_ARROW_CURSOR))
                {
                    cursorId = GLFW_ARROW_CURSOR;
                    gph::Input::setCursor(cursorId);
                }
                if(ImGui::Selectable("CROSSHAIR", cursorId == GLFW_CROSSHAIR_CURSOR))
                {
                    cursorId = GLFW_CROSSHAIR_CURSOR;
                    gph::Input::setCursor(cursorId);
                }
                if(ImGui::Selectable("HAND", cursorId == GLFW_HAND_CURSOR))
                {
                    cursorId = GLFW_HAND_CURSOR;
                    gph::Input::setCursor(cursorId);
                }

                ImGui::EndCombo();
            }

            if(secondWindow)
            {
                if(ImGui::Button("Close Second Window"))
                    secondWindow->shouldClose();
            } else
            {
                if(ImGui::Button("Open Second Window"))
                    secondWindow.reset(new gph::Window(300, 100, "secondWindow"));
            }

            if(ImGui::Button("Close App"))
                window.shouldClose();

            ImGui::Separator();

            ImGui::Text("Main window clipboard");

            static std::string clipboard;
            ImGui::InputText("##clipboardInput", &clipboard);

            if(ImGui::Button("Copy to Clipboard"))
                gph::Input::setClipboard(clipboard);

            ImGui::Text("content: %s", gph::Input::getClipboard().c_str());
        }
        ImGui::End();



        // show demo window
        if(show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);


        if( secondWindow )
        {
            if(secondWindow->update(false))
            {
                // we could render stuff in the second window
            } else
            {
                // close the second window
                secondWindow.reset();
            }
        }
    }

    return 0;
}
