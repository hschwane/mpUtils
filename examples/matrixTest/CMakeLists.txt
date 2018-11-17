cmake_minimum_required(VERSION 3.8)

# create target
add_executable(matrixTest main.cu)

# link libraries
target_link_libraries(matrixTest mpUtils::mpUtils)

