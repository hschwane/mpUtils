# my todo list

## high priority

- clean up cuda includes

## medium priority

### cmake
- https://cmake.org/pipermail/cmake/2017-December/066768.html

### finish the input manager
- print used keys
- functions to change keys afterwards
- allow to trigger keys from the code
- some default / common used hotkeys

### imgui extensions
- add hotkey support
- include usefull imgui addons
- write some good windows, ie logging, performance display, changing keys

### 3d rendering
- add some geometric primitives

### 2d renderer
- add 2d camera
- add a rendering engine for sprites and animated sprites

### graphics in general
- add double / triple buffering helper

### bit manipulation
- add macros or constexpr functions for bit manipulation

### bugfix
- make timers use a steady clock

### math
- better floating point comparism modes

## low priority / ideas / long term projects
- better + safer handling of uniforms for shaders
- add utility to parse command line input
- add networking capabilities
- add serial port capabilities
- timer set for profiling apps
- jason and/or xml and easy serializing
- a argument parser for command line apps
- util for handling general settings
- tests
- improvements to the log (lock-free-queue, more sinks with better format, different depths for messages)
- support for different ogl versions / have a manager to check for versions and extensions
- maybe add an openGL state / context wrapper?
- add texture compression and anisotropic filtering
- some spline support
- add text rendering
- add a proper 3d scene renderer
- more other graphics related features

# finished
for motivation, all finished todo entries are moved here instead of being deleted

--- v0.12.0 -- 20.12.2019 
- add wrapper to map a openGL buffer to cuda
- add cuda device vector
- add access functions to buffer (mirroring std::vector)
- add vector reference to access vectors managing gpu memory from the gpu
- add allocators for using managed and pinned memory in std vectors
- add missing references to third party software
- make openGL buffer copyable
- addBlockStrideRange for cuda utils
- various bux fixes (camera, texture, buffer, ...)

--- v0.11.1 --- 21.10.2019
- various bug fixes including the vertexArrayObject

--- v0.11.0 --- 15.10.2019
- add new hybrid (trackball/fps) camera
- add imgui loading animations
- add tinyfiledialogs
- add 2d transform 
- add wrapper for sync object
- rewrite / improve / repair opengl wrapper
- fix shader asserts
- remove handle class since it does not make sense
- better openGL error checking
- shader defines are automatically added depending on current shader stage
- remove all old or unusable graphics functions
