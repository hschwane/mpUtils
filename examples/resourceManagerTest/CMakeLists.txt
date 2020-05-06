cmake_minimum_required(VERSION 3.8)

# create target
add_executable(resMngrTest main.cpp)

# set required language standard
set_target_properties(resMngrTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED YES
        )

# link libraries
target_link_libraries(resMngrTest mpUtils::mpUtils)

