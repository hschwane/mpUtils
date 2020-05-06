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
- add a rendering animated sprites
- add 2d particle systems
- add 2d lighting and shadows
- add better sprite management

### bugfix
- make timers use a steady clock

## low priority / ideas / long term projects

- add double / triple buffering helper
- better + safer handling of uniforms for shaders
- add utility to parse command line input
- add networking capabilities
- add serial port capabilities
- timer set for profiling apps
- jason and/or xml and easy serializing
- util for handling general settings
- tests
- profiling / visual profiling tools
- improvements to the log (lock-free-queue, more sinks with better format, different depths for messages)
- support for different ogl versions / have a manager to check for versions and extensions
- maybe add an openGL state / context wrapper?
- add texture compression and anisotropic filtering
- some spline support
- add text rendering
- add a proper 3d scene renderer
- more other graphics related features
- better floating point comparism modes
- add macros or constexpr functions for bit manipulation



# finished
for motivation, all finished todo entries are moved here instead of being deleted

- add resource manager
- rename incbin resource to embedded data
- replace cfg handler with toml11
- add image class for image loading and manipulation
- add CallbackHandler for all classes that provide callbacks
- add simple imgui popups for messages
- add texture to imgui helper function
- add state machine class
- fix window class const correctness
- better sprite rendering
- add 2d camera
- fix bugs for opengl buffer
- fix bugs in string utils
- add cuda memory transfer helper functions
- various bugs fixed in the new cuda memory management
- add cuda copy helper functions
- add compare operators to helper math
- fix bugs in cuda vectors

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
