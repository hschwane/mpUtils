cmake_minimum_required(VERSION 3.8)

# create target
add_executable(GLexample1 main.cpp)

# set required language standard
set_target_properties(GLexample1 PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )

# link libraries
target_link_libraries(GLexample1 mpUtils::mpUtils)

