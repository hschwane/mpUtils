cmake_minimum_required(VERSION 3.8)


#add the resource
#include(addResources)
#ADD_RESOURCES(testResources blubber/hw.txt)

# create target
add_executable(resourceCompiler main.cpp ${testResources})

# set required language standard
set_target_properties(resourceCompiler PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )

target_compile_options(resourceCompiler PRIVATE "-Wa,-I${CMAKE_CURRENT_LIST_DIR}")

# link libraries
target_link_libraries(resourceCompiler mpUtils::mpUtils)


