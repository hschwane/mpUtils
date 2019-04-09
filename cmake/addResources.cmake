# found 08.04.2019 on https://beesbuzz.biz/code/4399-Embedding-binary-resources-with-CMake-and-C-11

FUNCTION(ADD_RESOURCES out_var)
    SET(result)
    FOREACH(in_f ${ARGN})
        SET(out_f "${CMAKE_CURRENT_BINARY_DIR}/${in_f}.o")
        get_filename_component(out_dir ${out_f} DIRECTORY)
        FILE(MAKE_DIRECTORY ${out_dir})
        ADD_CUSTOM_COMMAND(OUTPUT ${out_f}
                COMMAND ld -r -b binary -o ${out_f} ${in_f}
                DEPENDS ${in_f}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Building resource object ${out_f}"
                VERBATIM
                )
        LIST(APPEND result ${out_f})
        message("Added resource: ${CMAKE_CURRENT_LIST_DIR}/${in_f} as ${in_f}")
    ENDFOREACH()
    SET(${out_var} "${result}" PARENT_SCOPE)
ENDFUNCTION()