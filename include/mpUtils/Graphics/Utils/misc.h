/*
 * mpUtils
 * misc.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MISC_H
#define MPUTILS_MISC_H

// includes
//--------------------
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "mpUtils/Log/Log.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
// some global functions for the graphics framework

/**
 * Print some info about the supported openGL version to the log
 */
void inline logGlIinfo()
{
    logINFO("Graphics") << "Printing openGL version information:"
                        << "\n\t\tOpenGL version: " << glGetString(GL_VERSION)
                        << "\n\t\tGLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION)
                        << "\n\t\tVendor: " << glGetString(GL_VENDOR)
                        << "\n\t\tRenderer: " << glGetString(GL_RENDERER)
                        << "\n\t\tGLFW. Version: " << glfwGetVersionString();
}

/**
 * pass "true" to enable or "false" to disable Vsync
 */
void inline enableVsync(bool enabled)
{
    if(enabled)
        glfwSwapInterval(1);
    else
        glfwSwapInterval(0);
}

/** Calculates the byte offset of a given member.
 * usage:
 * auto off = offset_of(&MyStruct::my_member);
 */
template<typename T, typename TMember>
GLuint offset_of(TMember T::* field) noexcept
{
    // Use 0 instead of nullptr to prohibit a reinterpret_cast of nullptr_t
    // which throws a compiler error on some compilers.
    return static_cast<GLuint>(reinterpret_cast<size_t>(&(reinterpret_cast<T*>(0)->*field)));
}

/**
 * @brief transforms a cursor position into a world position in 2d
 * @param mouse the mose position as returned by window
 * @param viewport the viewport upper left corner and size
 * @param viewProjection the viewProjection matrix used to render the image
 * @return cursor position in world coordinates
 */
inline glm::vec2 mouseToWorld2D(const glm::ivec2& mouse, const glm::ivec4& viewport, const glm::mat4& viewProjection)
{
    glm::vec2 normMouse = -1.0f + (((glm::vec2(mouse) - glm::vec2(viewport.x,viewport.y)) / glm::vec2(viewport.z,viewport.w)) * 2.0f);
    normMouse.y = -normMouse.y;
    glm::vec4 projectedMouse =  glm::inverse(viewProjection) * glm::vec4(normMouse,0,1);
    logINFO("") << glm::to_string(viewProjection);
    logINFO("") << glm::to_string(glm::inverse(viewProjection));
    return glm::vec2(projectedMouse) / projectedMouse.w;
}


}}

#endif //MPUTILS_MISC_H
