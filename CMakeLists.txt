# -------------------------------------------------------------
# general setup
# -------------------------------------------------------------
cmake_minimum_required(VERSION 3.8...3.12)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# Appends the cmake/ path to MAKE_MODULE_PATH variable.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# create project
include(GetVersionFromGit)
project(mpUtils VERSION "${VERSION_SHORT}" LANGUAGES CXX C)

# default build configuration
include(setDefaultTypeRelease)


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
option(DISABLE_PATHS "Disable automatic inclusion of paths.h. (Note: this is also possible on a per project bases by defining MPU_NO_PATHS before including mpUtils.h)" OFF)
option(BUILD_EXAMPLES "Whether or not examples should be build." OFF)
option(EXPORT_BUILD_TREE "Enable, to export the targets from the build tree to your cmake registry. Useful for development. Use Together with FORCE_NEW_VERSION" OFF)
option(BUILD_SHARED_LIBS "Build a shared library (.so/.dll) instead of a static one-" ON)

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
add_library(mpUtils "")
add_library(mpCudaSettings INTERFACE)

# add an alias so that library can be used inside the build tree, e.g. when testing
add_library(mpUtils::mpUtils ALIAS mpUtils)
add_library(mpUtils::mpCudaSettings ALIAS mpCudaSettings)

# add required source files
target_sources(mpUtils PRIVATE
                "src/Misc/stringUtils.cpp"
                "src/Log/LogStream.cpp"
                "src/Log/FileSink.cpp"
                "src/Log/ConsoleSink.cpp"
                "src/Log/BufferedSink.cpp"
                #"src/Log/SyslogSink.cpp"
                "src/Log/Log.cpp"
                "external/snippets/src/stb_image_impl.cpp"
                "external/snippets/src/stb_image_write_impl.cpp"
                "external/snippets/src/tinyfd/tinyfiledialogs.c"
                "src/ResourceManager/mpUtilsResources.cpp"
                "src/ResourceManager/ResourceCache.cpp"
                "src/Misc/Image.cpp"
              )

# add optional source files
if (USE_OPENGL)
    target_sources(mpUtils PRIVATE
                    "src/Graphics/Opengl/Framebuffer.cpp"
                    "src/Graphics/Opengl/Shader.cpp"
                    "src/Graphics/Opengl/Texture.cpp"
                    "src/Graphics/Opengl/Sampler.cpp"
                    "src/Graphics/Opengl/Buffer.cpp"
                    "src/Graphics/Opengl/SyncObject.cpp"
                    "src/Graphics/Rendering/Camera.cpp"
                    "src/Graphics/Rendering2D/Camera2D.cpp"
                    "src/Graphics/Rendering2D/Sprite2D.cpp"
                    "src/Graphics/Rendering2D/Renderer2D.cpp"
                    "src/Graphics/Utils/Transform.cpp"
                    "src/Graphics/Utils/Transform2D.cpp"
                    "src/Graphics/Window.cpp"
                    "external/snippets/src/imgui/imgui.cpp"
                    "external/snippets/src/imgui/imgui_demo.cpp"
                    "external/snippets/src/imgui/imgui_draw.cpp"
                    "external/snippets/src/imgui/imgui_widgets.cpp"
                    "external/snippets/src/imgui/impl/imgui_impl_opengl3.cpp"
                    "external/snippets/src/imgui/impl/imgui_impl_glfw.cpp"
                    "external/snippets/src/imgui/stdlib/imgui_stdlib.cpp"
                    "src/Graphics/Gui/ImGui.cpp"
                    "src/Graphics/Gui/ImGuiStyles.cpp"
                    "src/Graphics/Gui/ImGuiElements.cpp"
                    "src/Graphics/Gui/ImGuiWindows.cpp"
                    "src/Graphics/Input.cpp"
                    "src/ResourceManager/mpGraphicsResources.cpp"
                  )
endif()


# -------------------------------------------------------------
# set include dirs
# -------------------------------------------------------------
target_include_directories(mpUtils
                            PUBLIC
                                $<INSTALL_INTERFACE:include>
                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/snippets/include>
                            PRIVATE
                                ${CMAKE_CURRENT_SOURCE_DIR}/src
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/snippets/include/mpUtils/external/imgui
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/snippets/src/imgui/impl
                                ${CMAKE_CURRENT_SOURCE_DIR}/external/snippets/include/mpUtils/external/imgui
        )


# -------------------------------------------------------------
# set target properties
# -------------------------------------------------------------

# set version and debug postfix
set_target_properties(mpUtils PROPERTIES
                        VERSION     ${VERSION_SHORT}
                        SOVERSION   ${VERSION_MAJOR}
                        DEBUG_POSTFIX "d"
                     )

# set required language standard
set_target_properties(mpUtils PROPERTIES
                        CXX_STANDARD 14
                        CXX_STANDARD_REQUIRED YES
                        CUDA_STANDARD 14
                        CUDA_STANDARD_REQUIRED YES
                     )

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
if(DISABLE_PATHS)
    target_compile_definitions(mpUtils PUBLIC MPU_NO_PATHS)
endif()

# defines for stb image
target_compile_definitions(mpUtils PUBLIC STBI_FAILURE_USERMSG)

# generate include files for version and paths
set(MPU_CUDA_AVAILIBLE ${USE_CUDA})
set(MPU_GLM_AVAILIBLE ${USE_GLM})
set(MPU_OPENGL_AVAILIBLE ${USE_OPENGL})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/CmakeConfigureTemplates/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/mpUtils/version.h)

set(MPU_LIB_SHADER_PATH ${CMAKE_CURRENT_LIST_DIR}/shader/)
set(MPU_LIB_RESOURCE_PATH ${CMAKE_CURRENT_LIST_DIR}/resources/)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/CmakeConfigureTemplates/paths.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/mpUtils/paths.h)

target_compile_options(mpUtils PRIVATE "-Wno-comment")
target_compile_options(mpUtils PRIVATE "-Wno-dangling-else")
#target_compile_options(mpUtils PRIVATE "-Wa,-I${CMAKE_CURRENT_LIST_DIR}/resources")

# -------------------------------------------------------------
# link dependencies (this will also link the dependencies of dependencies and set required compiler flags)
# -------------------------------------------------------------

# make imported targets cuda safe
if(USE_CUDA)
#    target_link_libraries(mpUtils PUBLIC libcudart.so)
    include(cudaConvertFlags)
    cuda_convert_flags(Threads::Threads)
    if(USE_GLM)
        cuda_convert_flags(glm)
    endif()
    if(USE_OPENGL)
        cuda_convert_flags(OpenGL::GL)
        cuda_convert_flags(glfw)
        cuda_convert_flags(GLEW::GLEW)
    endif()
endif()

# link required external libs
target_link_libraries(mpUtils PUBLIC Threads::Threads)

# link optional external libs
if(USE_GLM)
    target_link_libraries(mpUtils PUBLIC glm)
endif()

if(USE_OPENGL)
    target_link_libraries(mpUtils PUBLIC OpenGL::GL glfw GLEW::GLEW)
endif()

# link windows specific stuff needed for tinyfiledialogs
if(WIN32)
    target_link_libraries(mpUtils PUBLIC Comdlg32.lib Ole32.lib)
endif()

# link unix specific stuff
if(UNIX)
    target_link_libraries(mpUtils PUBLIC stdc++fs)
endif()


# --------------------------------------------------------
# build and link dependencies in external/modules
# --------------------------------------------------------
list(APPEND LOCAL_EXPOR_TARGETS "mpUtils")
if(USE_OPENGL)
    add_subdirectory(external/modules/GLshader)
    target_link_libraries(mpUtils PUBLIC glsp::glsp)
    list(APPEND LOCAL_EXPOR_TARGETS "glsp")
endif()


# --------------------------------------------------------
# installing and exporting
# --------------------------------------------------------
set_target_properties(mpUtils PROPERTIES EXPORT_NAME mpUtils)
set_target_properties(mpCudaSettings PROPERTIES EXPORT_NAME mpCudaSettings)

# set install dirs
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/mpUtils)

# configure the paths.h file to point to the installes shaders and resources
install(CODE "
        set(MPU_LIB_SHADER_PATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/mpUtils/shader)
        set(MPU_LIB_RESOURCE_PATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/mpUtils/resources)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/CmakeConfigureTemplates/paths.h.in
                        ${CMAKE_CURRENT_BINARY_DIR}/include/installedPaths.h)
        ")

# install binaries
install(TARGETS mpUtils mpCudaSettings
        EXPORT mpUtils-targets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

# install headers
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/snippets/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/mpUtils/version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mpUtils/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/installedPaths.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mpUtils/ RENAME paths.h)

# install usefull cmake scripts
install(FILES
        ${CMAKE_CURRENT_LIST_DIR}/cmake/selectComputeArch.cmake
        ${CMAKE_CURRENT_LIST_DIR}/cmake/cudaConvertFlags.cmake
        ${CMAKE_CURRENT_LIST_DIR}/cmake/GetGitRevisionDescription.cmake
        ${CMAKE_CURRENT_LIST_DIR}/cmake/GetGitRevisionDescription.cmake.in
        ${CMAKE_CURRENT_LIST_DIR}/cmake/GetVersionFromGit.cmake
        ${CMAKE_CURRENT_LIST_DIR}/cmake/setDefaultTypeRelease.cmake
        DESTINATION ${INSTALL_CONFIGDIR}/modules
        )

# install shader and resources
install(DIRECTORY shader DESTINATION ${CMAKE_INSTALL_DATADIR}/mpUtils/)
install(DIRECTORY resources DESTINATION ${CMAKE_INSTALL_DATADIR}/mpUtils/)

# install the Targets.cmake file
install(EXPORT mpUtils-targets
        FILE mpUtilsTargets.cmake
        NAMESPACE mpUtils::
        DESTINATION ${INSTALL_CONFIGDIR}
        )

# Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/mpUtilsConfigVersion.cmake
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY ExactVersion
                                )

# configure the mpUtilsConfig.cmake
set(ORIGINAL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(ORIGINAL_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/mpUtilsConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/mpUtilsConfig.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
        )

# Install the config and configversion
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/mpUtilsConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/mpUtilsConfigVersion.cmake
        DESTINATION ${INSTALL_CONFIGDIR}
        )


# --------------------------------------------------------
# export to build tree to allow instant use
# --------------------------------------------------------

# export targets from build tree
export(TARGETS ${LOCAL_EXPOR_TARGETS} NAMESPACE mpUtils:: FILE ${CMAKE_CURRENT_BINARY_DIR}/mpUtilsTargets.cmake)

if(EXPORT_BUILD_TREE)
#     register package in users package registry
    export(PACKAGE mpUtils)
endif()


# --------------------------------------------------------
# see if there are executables in the examples folder and add the subdirectories
# --------------------------------------------------------
if(BUILD_EXAMPLES)
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
else()
    message(STATUS "Not building examples. Set BUILD_EXAMPLES to build examples.")
endif()
