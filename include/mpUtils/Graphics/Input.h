/*
 * mpUtils
 * Input.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 * Defines public functions of the input manager.
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

/**
 * @brief Possible types (subclasses) on an InputFunction
 */
enum class InputFunctionType
{
    button, //!< the input function is a button which can have different behaviors (see ButtonBehavior)
    axis //!< the input is an axis and float values will be passed to it
};

/**
 * @brief defines when a button event is triggered
 */
enum class ButtonBehavior
{
    onPress, //!< input function will be called when the button is pressed
    onRelease, //!< input function will be called when the button is released
    onRepeat, //!< input function will be called when the button is repeated
    onDoubleClick, //!< input function is only called after button is pressed, released and pressed again. time can be configured with
    onPressRepeat, //!< input function is called when button is pressed and then again when it is repeated
    whenDown,    //!< function is called every frame that button is down like when polling isButtonDown()
    defaultBehavior //! use the default button behavior instead of the overwritten one (only to be used when mapping input)
};

/**
 * @brief Controls what happens when a is mapped to a digital input or a button to a analog input
 */
enum class AxisBehavior
{
    positive = 1, //!< positive analog input will trigger button / digital input will move axis in positive direction
    negative = -1, //!< negative analog input will trigger button / digital input will move axis in negative direction
    defaultBehavior =0 //!< no button axis interaction (!! means no interaction between analog input with button / digital input with axis!!)
};

/**
 * @brief switch between horizontal and vertical axis
 */
enum class AxisOrientation
{
    horizontal, //!< horizontal axis if availible
    vertical //!< vertical axis (default)
};

// polling
bool isKeyDown(int key);
bool isMouseButtonDown(int key);
std::pair<Window*,glm::dvec2> getCursorPos();
glm::dvec2 getCursorScreenPos();
Window* getActiveWindow();
Window* getHoveredWindow();

// control the cursor
void setCursorScreenPos(double x, double y);
void setCursorScreenPos(glm::dvec2 p);
void setCursor(GLFWcursor* c);
void setCursor(int shape);

// deal with the clipboard
std::string getClipboard();
void setClipboard(const std::string& text);

// add and remove custom callbacks
int addDropCallback(std::function<void(Window&, const std::vector<std::string>&)> f);
void removeDropCallback(int id);

// change global settings
void setDoubleClickTime(unsigned int ms);
unsigned int getDoubleClickTime();
void setAnalogToDigitalRatio(double r);
double setAnalogToDigitalRatio();
void setDigitaltoAnalogRatio(double r);
double setDigitalToAnalogRatio();


void addButton(std::string name, std::string description, std::function<void(Window&)> function,
               ButtonBehavior behavior = ButtonBehavior::onPress, bool allowBehaviorOverride = true, bool active=true);
void addAxis(std::string name, std::string description, std::function<void(Window&,double)> function, bool active=true);


void mapKeyToInput(std::string name, int key, int requiredMods = 0,
            ButtonBehavior overrideBehavior = ButtonBehavior::defaultBehavior, AxisBehavior ab = AxisBehavior::defaultBehavior);
void mapMouseButtonToInput(std::string name, int button, int requiredMods = 0,
                           ButtonBehavior overrideBehavior = ButtonBehavior::defaultBehavior,
                           AxisBehavior ab = AxisBehavior::defaultBehavior);
void mapScrollToInput(std::string name, int requiredMods = 0, AxisBehavior direction = AxisBehavior::defaultBehavior,
                      AxisOrientation axis = AxisOrientation::vertical);
void mapCourserToInput(std::string name, AxisOrientation axis, int requiredMods = 0, AxisBehavior direction = AxisBehavior::defaultBehavior);

}}}
#endif //MPUTILS_INPUT_H
