cmake_minimum_required(VERSION 3.18)

if(NOT USE_CUDA)
    return()
endif()

# create target
add_executable(cudaTest main.cu)

# set required language standard
set_target_properties(cudaTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        )

# link libraries
target_link_libraries(cudaTest mpUtils::mpUtils)

