cmake_minimum_required(VERSION 3.8)

# create target
add_executable(templateExperiments main.cpp)

# set required language standard
set_target_properties(templateExperiments PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED YES
        )

#target_compile_options(templateExperiments PRIVATE --expt-relaxed-constexpr)

# link libraries
target_link_libraries(templateExperiments mpUtils::mpUtils)


