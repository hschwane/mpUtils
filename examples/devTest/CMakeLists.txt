cmake_minimum_required(VERSION 3.8)

# create target
add_executable(devTest main.cpp)

# link libraries
target_link_libraries(devTest mpUtils::mpUtils)

