# mpUtils
A ~~platform independent~~ linux c++ library with some utilities I use in a lot of my projects. It also features support
for computer graphics applications using openGL and CUDA.

-------------------------

## installation

To use mpUtils you need to build and install it yourself from sources.

### dependencies

To install mpUtils you will need:
- CMake version 3.8 or higher
- a c++ 14 compatible compiler, eg gcc5+

To use the graphics functionality you will also need:
- glm
- glfw3
- glew
- graphics drivers supporting some OpenGL version (preferably 4.5+)

To use the cuda functionality you will need:
- CUDA-toolkit version 9+

###### HINT
If some of the dependencies are missing mpUtils will build and install fine, but give you a warning.
In this case the corresponding features will not be availible to applications.

### build and install

After installing dependencies for the features you need, the following commands will download, build and install
the newest version of mpUtils into the default directory:

```
git clone https://github.com/hschwane/mpUtils.git
cd mpUtils
mkdir build
cd buils
cmake ..
make -j 8 #<change 8 to the number of procesors you want to use for compiling>
sudo make install
```

If you need a specific version, please manually download it from the github page.
To change the installation directory (e.g. to a directory where you have write permissions)
use the cmake variable `CMAKE_INSTALL_PREFIX`.

###### HINT
Do not forget to set `CMAKE_PREFIX_PATH` or `mpUtills_DIR` in the projects that use mpUtils if you changed the installation directory.

### cmake options

There is a number of options availible when building mpUtils.
You can set the corresponding cmake variables by directly passing them to the cmake call on the commandline or by using a cmake gui.
It is also possible to use the console (curse-)gui `ccmake` or to manually modify the generated CMakeCache.txt file and then calling `cmake ..` again.
The following options are availible:
- `CMAKE_BUILD_TYPE` change the build type, Release is the default.
- `CMAKE_INSTALL_PREFIX` overwrite the default installation directory.
- `USE_GLM` if enabled GLM functionality will be availible to projects linking to mpUtils.
- `USE_OPENGL` if enabled openGL and graphics functionality will be availible to projects linking to mpUtils.
- `USE_CUDA` if enabled CUDA functionality will be availible to projects linking to mpUtils.
- `CUDA_ARCH` the CUDA version for which kernel code is compiled. Default is Auto and tries to detect your graphics card.
                Alternatively you can specify one or more Architectures (eg Volta+Pascal) or compute capability version numbers (eg 5.2).
                You can also use All to generate code for all compatible GPUs.
- `DISABLE_PPUTILS` by default mpUtils defines and includes a range of preprocessor utilities.
                        If enabled Preprocessor utils will not be defined. You can also disable the defines on a per project basis by
                        defining `MPU_NO_PREPROCESSOR_UTILS` before inclusion of mpUtils.h
- `DISABLE_PATHS` by default mpUtils defines its paths to its shader and resource folders.
                       If this option is enabled the paths will not be defined. You can also disable the defines on a per project basis by
                       defining `MPU_NO_PATHS` before inclusion of mpUtils.h and mpGraphics.h.
- `BUILD_EXAMPLES` if enabled, the examples folder is searched for executables and they are also compiled.
- `EXPORT_BUILD_TREE` if enabled, the build tree will be added to your cmake package registry. That enables other projects to find the library without installing it.
                        mainly used during development and testing. Consider using FORCE_NEW_VERSION as well.
- `FORCE_NEW_VERSION` if enabled a instead of using the version number of the git commit, the version is increased by one.
                        This is mainly used during development and testing to not screw with existing mpUtils installations
                        on the development system. Often used together with EXPORT_BUILD_TREE.

--------------------------
## usage

To use mpUtils in a project use cmake find_package and target_link_libraries:
``` cmake
project(demo CXX)
add_executable(demo main.cpp)

find_package(mpUtills REQUIRED)
target_link_libraries(demo PUBLIC mpUtils::mpUtils)
```

when mpUtils is found the following targets will be defined:
- `mpUtils::mpUtils` linking this target will add all settings needed to build a app with mpUtils.
                        Including compiler settings, include directories and linker settings.
- `mpUtils::cudaSettings` linking this target will set settings for CUDA compilation such as the the Architectures for which code
                            should be generated.

in addition the following variables will be set:
- `mpUtils_FOUND` whether or not mpUtils was found.
- `mpUtils_CMAKE_DIR` the directory where the mpUtils installation is located.
- `mpUtils_VERSION` the version of the mpUtils installation.
- `mpUtils_CMAKE_SCRIPTS_PATH` this path can be added to the cmake module path to have access to some useful cmake tools (see the cmake subfolder).
- `mpUtils_GLM_AVAILIBLE`
- `mpUtils_OPENGL_AVAILIBLE`
- `mpUtils_CUDA_AVAILIBLE`
- `mpUtils_PPUTILS_AVAILIBLE`
- `mpUtils_CUDA_ARCH_FLAGS` only if `mpUtils_CUDA_AVAILIBLE` is true. These are the cuda architecture flags selected during mpUtils installation.

Now, to actually use mpUtils functionality include the file `mpUtils/mpUtils.h`.
Most classes and functions live in the namespace `mpu::`. To use the graphics utils include the file `mpGraphics`.
The graphics utils are in the namespace `mpu::gph::`.

To use the CUDA utilities you will need to call `enable_language(CUDA)` in your CmakeList.txt then from your .cu files
include the `mpCuda.h` header to gain access to all the CUDA specific utilities. Most of the things from `mpUtils.h` can
also be used in cuda code. Some even in device functions.

--------------------------
## features

- c++ wrapper for most openGl and glfw functionality
- some utilities for graphics applications like camera handling and screen filling rendering
- utilities for glsl shader like a c-style preprocessor as well as include files with math functions, rng and noise
- building with cuda, including cuda support for the clion IDE and some cuda helping functions and utilities
- vector and matrix math functions for cuda
- cmake modules for handling versions and cuda code generation
- highly customisable thread safe logger
- ini-Style configuration file parser
- different timers including asynchronous ones
- doxygen style documentation
- includes compile-time math functions from the gcem-library (see 3rd party code)
- compile and embed resources into executables

------------------------
## planned features / wishlist

- more utilities for graphics programming (maybe game engine style functionality)
- utils for handling input better
- include a GUI
- network communication
- serial communication
- jason and/or xml and easy serializing
- a argument parser for command line apps
- util for handling general settings
- tests
- windows and mac support
- improvements to the log (lock-free-queue, more sinks with better format, different depths for messages)

-----------------------
## included 3rd party code

GCE-Math: A C++ generalized constant expression-based math library
Copyright 2016-2019 Keith O'Hara
This product includes software developed by Keith O'Hara (http://www.kthohr.com)

This software contains source code
provided by NVIDIA Corporation.

stb_image (https://github.com/nothings/stb)
This software contains source code provided by Sean T. Barrett.

Dear ImGui (https://github.com/ocornut/imgui)
This software contains source code provided by Omar Cornut.