# -------------------------------------------------------------
# general setup
# -------------------------------------------------------------
cmake_minimum_required(VERSION 3.8)

# Appends the cmake/ path to MAKE_MODULE_PATH variable.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# create project
include(GetVersionFromGit)
project(mpUtils VERSION "${VERSION_SHORT}" LANGUAGES CXX)


# -------------------------------------------------------------
# try to find dependencies
# -------------------------------------------------------------

# find required packages
find_package(Threads REQUIRED)

# find optional packages
find_package(glm QUIET)
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL QUIET)
find_package(GLEW QUIET)
find_package(glfw3 QUIET)

# check if cuda is availible
include(CheckLanguage)
check_language(CUDA)


# -------------------------------------------------------------
# user settings and optional features
# -------------------------------------------------------------

# possibility to turn of features with special dependencies
include(CMakeDependentOption)

# make cache entries for optional features
cmake_dependent_option(USE_CUDA "Compile with cuda features." ON "CMAKE_CUDA_COMPILER" OFF)
cmake_dependent_option(USE_GLM "Compile with GLM. This is required for openGL." ON "glm_FOUND" OFF)
cmake_dependent_option(USE_OPENGL "Compile with openGL/graphics features." ON "OPENGL_FOUND; glm_FOUND; GLEW_FOUND; glfw3_FOUND" OFF)
option(DISABLE_PPUTILS "Disable inclusion of preprocessor utils. (Note: this is also possible on a per project bases by defining MPU_NO_PREPROCESSOR_UTILS before including mpUtils.h)" OFF)

# dependencies between features
if(USE_OPENGL AND (NOT USE_GLM))
    set(USE_GLM ON FORCE)
    message(WARNING "GLM is required for openGL support! Enabling GLM...")
endif()

# check again to make sure nothing went wrong
if(NOT CMAKE_CUDA_COMPILER)
    if(USE_CUDA)
        message(SEND_ERROR "Cuda option was selected but cuda could not be found!")
    else()
        message(STATUS "Cuda was not found on found on your system. Cuda support is disabled.")
    endif()
endif()

if(NOT glm_FOUND)
    if(USE_GLM)
        message(SEND_ERROR "GLM option was selected but glm could not be found!")
    else()
        message(STATUS "GLM was not found on found on your system. GLM support is disabled.")
    endif()
endif()

if(NOT OPENGL_FOUND)
    if(USE_OPENGL)
        message(SEND_ERROR "OpenGL option was selected but OpenGL could not be found!")
    else()
        message(STATUS "OpenGL was not found on found on your system. OpenGL / graphics support is disabled.")
    endif()
endif()

if(NOT GLEW_FOUND)
    if(USE_OPENGL)
        message(SEND_ERROR "OpenGL option was selected but GLEW could not be found!")
    else()
        message(STATUS "GLEW was not found on found on your system. OpenGL / graphics support is disabled.")
    endif()
endif()

if(NOT glfw3_FOUND)
    if(USE_OPENGL)
        message(SEND_ERROR "OpenGL option was selected but glfw3 could not be found!")
    else()
        message(STATUS "GLFW3 was not found on found on your system. OpenGL / graphics support is disabled.")
    endif()
endif()


# -------------------------------------------------------------
# enable cuda
# -------------------------------------------------------------

if (USE_CUDA)
    enable_language(CUDA)

    include(selectComputeArch)
    set(CUDA_ARCH "Auto" CACHE STRING
            "Select cuda arches for which code is produced. Auto | Common | All | list of names | number")
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "${CUDA_ARCH}")
endif()


# -------------------------------------------------------------
# create targets and add sources
# -------------------------------------------------------------

# create target and add general cpp files
add_library(mpUtils SHARED "")
add_library(mpCudaSettings INTERFACE)

# add an alias so that library can be used inside the build tree, e.g. when testing
add_library(mpUtils::mpUtils ALIAS mpUtils)
add_library(mpUtils::mpCudaSettings ALIAS mpCudaSettings)

# add required source files
target_sources(mpUtils PRIVATE
                "../src/Misc/stringUtils.cpp"
                "../src/Log/LogStream.cpp"
                "../src/Log/FileSink.cpp"
                "../src/Log/ConsoleSink.cpp"
                "../src/Log/SyslogSink.cpp"
                "../src/Log/Log.cpp"
                "../src/Cfg/CfgFile.cpp"
                "../external/stb_image_impl.cpp"
              )

# add optional source files
if (USE_OPENGL)
    target_sources(mpUtils PRIVATE
                    "../src/Graphics/Rendering/Camera.cpp"
                    "../src/Graphics/Opengl/Framebuffer.cpp"
                    "../src/Graphics/Opengl/Shader.cpp"
                    "../src/Graphics/Opengl/Texture.cpp"
                    "../src/Graphics/Opengl/glsl/Preprocessor.cpp"
                    "../src/Graphics/Utils/Transform.cpp"
                    "../src/Graphics/Utils/ModelViewProjection.cpp"
                    "../src/Graphics/Window.cpp"
                  )
endif()

# -------------------------------------------------------------
# set include dirs
# -------------------------------------------------------------

target_include_directories(mpUtils
                            PUBLIC
                                $<INSTALL_INTERFACE:include>
                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            PRIVATE
                                ${CMAKE_CURRENT_SOURCE_DIR}/src
                        )


# -------------------------------------------------------------
# set target properties
# -------------------------------------------------------------

# set version
set_target_properties(mpUtils PROPERTIES
                        VERSION     ${VERSION_SHORT}
                        SOVERSION   ${VERSION_MAJOR}
                     )

# set required language standard
target_compile_features(mpUtils PUBLIC cxx_std_14)
set_target_properties(mpUtils PROPERTIES
                        CXX_STANDARD_REQUIRED YES
                        CUDA_STANDARD 14
                        CUDA_STANDARD_REQUIRED YES
                     )

# force c++17 if using opengl
if(USE_OPENGL)
    target_compile_features(mpUtils PUBLIC cxx_std_17)
endif()

# set cuda specific properties and compile options
if (USE_CUDA)
    set_target_properties( mpUtils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_compile_options( mpUtils PRIVATE
                                $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_ARCH_FLAGS}>
                                $<$<CONFIG:DEBUG>:$<$<COMPILE_LANGUAGE:CUDA>: -g -G>>
                            )
    target_compile_options( mpCudaSettings INTERFACE
                                $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_ARCH_FLAGS}>
                                $<$<CONFIG:DEBUG>:$<$<COMPILE_LANGUAGE:CUDA>: -g -G>>
                            )
endif ()

# set options for optional stuff
if(USE_GLM)
    target_compile_definitions(mpUtils PUBLIC GLM_ENABLE_EXPERIMENTAL)
endif()

if (USE_OPENGL)
    target_compile_definitions(mpUtils PUBLIC GLFW_INCLUDE_GLCOREARB)
    target_compile_definitions(mpUtils PUBLIC GLEW_NO_GLU)
endif()

if(DISABLE_PPUTILS)
    target_compile_definitions(mpUtils PUBLIC MPU_NO_PREPROCESSOR_UTILS)
endif()

# set definitions for pathes and library version
target_compile_definitions( mpUtils PUBLIC MPU_LIB_VERSION="${VERSION_SHORT}")

# TODO: make this compatible with installing
set(LIB_SHADER_PATH ${CMAKE_SOURCE_DIR}/shader CACHE PATH "Library Shader Path. Set manually if it was not found.")
set(LIB_RESOURCE_PATH ${CMAKE_SOURCE_DIR}/resources CACHE PATH "Library Resource path. Set manually if it was not found.")
add_definitions(-DLIB_SHADER_PATH="${LIB_SHADER_PATH}/")
add_definitions(-DLIB_RESOURCE_PATH="${LIB_RESOURCE_PATH}/")

# -------------------------------------------------------------
# link dependencies (this will also link the dependencies of dependencies and set required compiler flags)
# -------------------------------------------------------------

# make imported targets cuda safe
if(USE_CUDA)
    include(cudaConvertFlags)
    cuda_convert_flags(Threads::Threads)
    cuda_convert_flags(glm)
    cuda_convert_flags(OpenGL::GL)
    cuda_convert_flags(glfw)
    cuda_convert_flags(GLEW::GLEW)
endif()

# link required external libs
target_link_libraries(mpUtils PUBLIC stdc++fs Threads::Threads)

# link optional external libs
if(USE_GLM)
    target_link_libraries(mpUtils PUBLIC glm)
endif()

if(USE_OPENGL)
    target_link_libraries(mpUtils PUBLIC OpenGL::GL glfw GLEW::GLEW)
endif()


# ---------------------
# set install options
# ---------------------
# currently not supportet
# install(TARGETS mpUtils
#        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
#
#install(
#        DIRECTORY ${CMAKE_SOURCE_DIR}/src/
#        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mpUtils
#        FILES_MATCHING PATTERN "*.h*")

# --------------------------------------------------------
# see if there are executables and add the subdirectories
# --------------------------------------------------------

file(GLOB children RELATIVE ${CMAKE_SOURCE_DIR}/examples ${CMAKE_SOURCE_DIR}/examples/*)
set(subdirs "")
foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/examples/${child})
        if (NOT ${child} MATCHES "\\..*")
            if(EXISTS ${CMAKE_SOURCE_DIR}/examples/${child}/CMakeLists.txt)
                string(REPLACE " " "_" child ${child})
                set(subdirs ${subdirs} ${child})
                message("Found Executable in folder '${child}'.")
            endif()
        endif()
    endif()
endforeach()
foreach(n ${subdirs})
    add_subdirectory(${CMAKE_SOURCE_DIR}/examples/${n})
endforeach()
