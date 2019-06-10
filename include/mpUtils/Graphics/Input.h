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

enum class InputFunctionType
{
    button, //!< the input function is a button which can have different behaviors (see ButtonBehavior)
    axis //!< the input is an axis and float values will be passed to it
};

enum class ButtonBehavior
{
    onPress, //!< input function will be called when the button is pressed
    onRelease, //!< input function will be called when the button is released
    onRepeat, //!< input function will be called when the button is repeated
    onDoubleClick, //!< input function is only called after button is pressed, released and pressed again. time can be configured with
    onPressRepeat, //!< input function is called when button is pressed and then again when it is repeated
    whenDown,    //!< function is called every frame that button is down like when polling isButtonDown()
    defaultBehavior //! use the default button behavior instead of the overwritten one (only to be used by an input mapping)
};

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

void bindKeyToInput(std::string name, int key, int requiredMods = 0, ButtonBehavior overrideBehavior = ButtonBehavior::defaultBehavior);
void addButton(std::string name, std::string description, std::function<void(Window&)> function,
               ButtonBehavior behavior = ButtonBehavior::onPress, bool allowBehaviorOverride = true, bool active=true);

}}}
#endif //MPUTILS_INPUT_H
