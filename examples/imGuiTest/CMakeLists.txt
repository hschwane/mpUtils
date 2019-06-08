cmake_minimum_required(VERSION 3.3)

# find directories
set(PROJECT_SHADER_PATH "${CMAKE_CURRENT_LIST_DIR}/shader")
set(PROJECT_RESOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/resources")

# create target
add_executable(imGuiTest main.cpp
                            "imgui_impl_glfw.cpp"
                            "imgui_impl_opengl3.cpp")

# set required language standard
set_target_properties(imGuiTest PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED YES
        )

# set defines
target_compile_definitions(imGuiTest PRIVATE -DPROJECT_SHADER_PATH="${PROJECT_SHADER_PATH}/")
target_compile_definitions(imGuiTest PRIVATE -DROJECT_RESOURCE_PATH="${PROJECT_RESOURCE_PATH}/")

# link libraries
target_link_libraries(imGuiTest mpUtils::mpUtils)

