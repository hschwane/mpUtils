cmake_minimum_required(VERSION 3.8)

# create target
add_executable(graphicsTest main.cpp)

# set required language standard
set_target_properties(graphicsTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED YES
        )

# link libraries
target_link_libraries(graphicsTest mpUtils::mpUtils)

