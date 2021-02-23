cmake_minimum_required(VERSION 3.8)

if(NOT USE_OPENGL)
    return()
endif()

# create target
add_executable(imageTest main.cpp)

# set required language standard
set_target_properties(imageTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )

# link libraries
target_link_libraries(imageTest mpUtils::mpUtils)

