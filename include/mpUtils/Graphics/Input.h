/*
 * mpUtils
 * Input.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_INPUT_H
#define MPUTILS_INPUT_H

// includes
//--------------------
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "mpUtils/Graphics/Window.h"
#include <tuple>
#include <string>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
namespace Input {
//--------------------

bool isKeyDown(int key);
bool isMouseButtonDown(int key);
std::pair<Window*,glm::ivec2> getCursorPos();
glm::dvec2 getCursorScreenPos();
void setCursorScreenPos(double x, double y);
void setCursorScreenPos(glm::dvec2 p);
Window* getActiveWindow();
Window* getHoveredWindow();
void setCursor(GLFWcursor* c);
void setCursor(int shape);
std::string getClipboard();
void setClipboard(const std::string& text);

}}}
#endif //MPUTILS_INPUT_H
