/*
 * mpUtils
 * InputDetail.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 * Defines function of the input manager that are only used internally by other mpUtils modules.
 *
 */
#ifndef MPUTILS_INPUTINTERNAL_H
#define MPUTILS_INPUTINTERNAL_H

// includes
//--------------------
#include "GLFW/glfw3.h"
//--------------------

// forward definitions
//--------------------
namespace mpu { namespace gph { class Window; }}
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
namespace Input {
//--------------------

void registerWindow(Window* wnd); //!< register a window to the input manager
void unregisterWindow(Window* wnd); //!< remove a window from the input manager

}}}

// include forward declared classes
//--------------------
#include "mpUtils/Graphics/Window.h"
//--------------------

#endif //MPUTILS_INPUTINTERNAL_H
