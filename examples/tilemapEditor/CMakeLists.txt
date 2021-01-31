cmake_minimum_required(VERSION 3.3)

# find directories
set(PROJECT_SHADER_PATH "${CMAKE_CURRENT_LIST_DIR}/shader")
set(PROJECT_RESOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/resources")
set(MAP_WORKDIR "${CMAKE_CURRENT_LIST_DIR}/defaultWorkdir")

# create target
add_executable(tilemapEditor main.cpp TilemapEditor.cpp)

# set required language standard
set_target_properties(tilemapEditor PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )

# set defines
target_compile_definitions(tilemapEditor PRIVATE -DPROJECT_SHADER_PATH="${PROJECT_SHADER_PATH}/")
target_compile_definitions(tilemapEditor PRIVATE -DPROJECT_RESOURCE_PATH="${PROJECT_RESOURCE_PATH}/")
target_compile_definitions(tilemapEditor PRIVATE -DMAP_WORKDIR="${MAP_WORKDIR}/")

# link libraries
target_link_libraries(tilemapEditor mpUtils::mpUtils)
