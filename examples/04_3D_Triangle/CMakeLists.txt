cmake_minimum_required(VERSION 3.8)

if(NOT USE_OPENGL)
    return()
endif()

# create target
add_executable(example4 main.cpp)

# set required language standard
set_target_properties(example3 PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )

target_compile_definitions(example4 PRIVATE -DPROJECT_SHADER_PATH="${CMAKE_CURRENT_LIST_DIR}/")

# link libraries
target_link_libraries(example4 mpUtils::mpUtils)

