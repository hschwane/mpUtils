cmake_minimum_required(VERSION 3.8)

# create target
add_executable(tomlTest main.cpp)

# set required language standard
set_target_properties(tomlTest PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED YES
        )

# link libraries
target_link_libraries(tomlTest mpUtils::mpUtils)

