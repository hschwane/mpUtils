cmake_minimum_required(VERSION 3.8)


#add the resource
include(addResources)
ADD_RESOURCES(testResources blubber/hw.txt)

# create target
add_executable(resourceCompiler main.cpp ${testResources})

# set required language standard
set_target_properties(resourceCompiler PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED YES
        )

# link libraries
target_link_libraries(resourceCompiler mpUtils::mpUtils)


