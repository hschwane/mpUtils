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

/**
 * @brief Input handling functionallity is implemented in this namespace.
 *
 * usage:
 * Call update() once a frame. This is where all events from the window manager are handled including input events.
 * Then you can use poling functions to check for key states as well as get information on the cursor position and
 * the currently active window. Please note that window specific polling is implemented in the window class as well as
 * changing a windows input mode.
 * There are also functions to set the global cursor position as well as deal with the window managers clipboard content.
 * A number of custom calllback functions is also available. You can add multiple different callbacks of the same type if you need to.
 *
 * The heart of the input namespace is however managed input. This allows you to set input functions independent of hardware and later
 * create mappings to different devices. This way you can allow the end user to change the keys used to control the program.
 * Two types of inputs are available: button and axis. Button has a function that will be executed once the button is triggered, while
 * an axis also has a value that represents the rate of change of an analog input.
 * However axis can also be mapped to digital keys as can buttons be mapped to analog inputs.
 * Currently keyboard, mouse button, scroll events and cursor movement are supported in mappings.
 * Use generateHelpString() to generate a string with all input functions and mappings for the users information.
 *
 * Changing mappings and the state of inputs might be implemented in the future. Feel free to do so yourself if you need it now ;)
 *
 */
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

// important, call once every frame:
void update(); //!< handle all callbacks and other input stuff. call once per frame

// polling
bool isKeyDown(int key); //!< returns true if key (glfw named keycode) is down
bool isMouseButtonDown(int button);    //!< returns true if button is down (glfw mouse button id)
std::pair<Window*,glm::dvec2> getCursorPos(); //!< returns the window the coursor is hovering and the cursor position relative to that window
glm::dvec2 getCursorScreenPos(); //!< get cursor position in screen coordinates
Window* getActiveWindow(); //!< returns the active window or nullptr if no window is active
Window* getHoveredWindow(); // returns to window under the cursor or nullptr if no window is under the cursor

// control the cursor
void setCursorScreenPos(double x, double y); //!< set the cursor position in screen coordinates
void setCursorScreenPos(glm::dvec2 p); //!< set the cursor position in screen coordinates
void setCursor(GLFWcursor* c); //!< set a custom cursor shape
void setCursor(int shape); //!< set a cursor shape from glfw's default cursors

// deal with the clipboard
std::string getClipboard(); //!< get the content of the clipboard might throw if content is empty/can not be read
void setClipboard(const std::string& text);

// add and remove custom callbacks
int addUpdateCallback(std::function<void()> f); //!< f will be called whenever the input manager performs an update (once per frame). the returned id can be used to remove the callback later
void removeUpdateCallback(int id); //!< removes the update callback with id "id"
int addDropCallback(std::function<void(Window&, const std::vector<std::string>&)> f); //!< f will be called whenever files are dropped onto the window via drag and drop
void removeDropCallback(int id); //!< removes a drop callback by its id
int addCursorEnterCallback(std::function<void(Window&,bool)> f); //!< f is called whenever the cursor adds or leaves an application window
void removeCursorEnterCallback(int id); //!< removes a cursor enter callback by it's id
int addCharCallback(std::function<void(Window&, unsigned int)> f); //!< use f to recive character input
void removeCharCallback(int id); //!< remove a char callback by its id

// change global settings
void setDoubleClickTime(unsigned int ms); //!< sets the max time between to key presses for them to be registered as a double click
unsigned int getDoubleClickTime(); //!< returns the max time between to key presses for them to be registered as a double click
void setAnalogToDigitalRatio(double r); //!< sets the axis value a analog input needs to exceed in order to trigger a button input
double getAnalogToDigitalRatio(); //!< returns the axis value a analog input needs to exceed in order to trigger a button input
void setDigitaltoAnalogRatio(double r); //!< sets the value that is applied to an axis input when a digital button is pressed
double getDigitalToAnalogRatio(); //!< returns the value that is applied to an axis input when a digital button is pressed

// managed input handling

/**
 * @brief Add a button type input to the input manager.
 * @param name The Name of the input by which it is identified. Must be unique. Characters after a double hash "##" will not be displayed in gui or help output.
 * @param description Description of the input will be displayed in gui and help output.
 * @param function The function to be called when the button is triggered. a reference to the window which handled the event will be passed along.
 * @param behavior Defines when the button should be triggered eg onPress or onRelease
 * @param allowBehaviorOverride true to allow the behavior to be overriden by a specificinput mappin. (Eg use onPress for normal key but onDoubleClick for mouse button)
 * @param active True, if the button should be enabled by default
 */
void addButton(std::string name, std::string description, std::function<void(Window&)> function,
               ButtonBehavior behavior = ButtonBehavior::onPress, bool allowBehaviorOverride = true, bool active=true);

/**
 * @brief Add a axis type input to the input manager.
 * @param name The Name of the input by which it is identified. Must be unique. Characters after a double hash "##" will not be displayed in gui or help output.
 * @param description Description of the input will be displayed in gui and help output.
 * @param function The function is called whenever the value of the axis changes. The rate of change will be passed as well as the window that handled the event.
 * @param active True, if the button should be enabled by default
 */
void addAxis(std::string name, std::string description, std::function<void(Window&,double)> function, bool active=true);

/**
 * @brief Map a keyboard key to a input function. If you map it to a button the button will be triggered according to its behavior.
 *          If you map it to a axis a value of getDigitalToAnalogRatio() is applied to the axis in the direction specified by "ab".
 *          You can add multiple mappings to an input or use the same key in multiple mappings.
 *          If you map a key to an axis you MUST set "ab" or it will have no effect.
 * @param name Name of the input this mapping should apply to.
 * @param key The glfw named key to use for the mapping.
 * @param requiredMods A bit-set of modifiers that need to be pressed alongside this key for the input to be triggered
 * @param overrideBehavior Try to override the behavior of a button if allowed
 * @param ab The direction in which change is applied by this mapping to an axis input.
 */
void mapKeyToInput(std::string name, int key, int requiredMods = 0,
            ButtonBehavior overrideBehavior = ButtonBehavior::defaultBehavior,
            AxisBehavior ab = AxisBehavior::defaultBehavior);

/**
 * @brief Map a mouse button to a input function. If you map it to a button the button will be triggered according to its behavior.
 *          If you map it to a axis a value of getDigitalToAnalogRatio() is applied to the axis in the direction specified by "ab".
 *          You can add multiple mappings to an input or use the same mouse buttons in multiple mappings.
 *          If you map a button to an axis you MUST set "ab" or it will have no effect.
 * @param name Name of the input this mapping should apply to.
 * @param button The glfw mouse button id to use for the mapping.
 * @param requiredMods A bit-set of modifiers that need to be pressed alongside this button for the input to be triggered
 * @param overrideBehavior Try to override the behavior of a button if allowed
 * @param ab The direction in which change is applied by this mapping to an axis input.
 */
void mapMouseButtonToInput(std::string name, int button, int requiredMods = 0,
                           ButtonBehavior overrideBehavior = ButtonBehavior::defaultBehavior,
                           AxisBehavior ab = AxisBehavior::defaultBehavior);
/**
 * @brief Map a scroll action to a input function. If you map it to an axis the rate of scrolling will be applied as rate of change
 *          to the axis. If you map it to a button, the button will be triggered once whenever a rate of change of getAnalogToDigitalRatio
 *          is exceeded in the direction defined by "ab". You can add multiple mappings to an input or use the scroll event in multiple mappings.
 *          If you map a scroll event to a button you MUST set "ab" or it will have no effect.
 * @param name Name of the input this mapping should apply to.
 * @param requiredMods A bit-set of modifiers that need to be pressed while the scroll event is recorded for the input to be triggered
 * @param ab The direction in which this needs to be moved in order to trigger a button input.
 * @param axis The orientation of the scroll event (default is vertical scrolling)
 */
void mapScrollToInput(std::string name, int requiredMods = 0, AxisBehavior direction = AxisBehavior::defaultBehavior,
                      AxisOrientation axis = AxisOrientation::vertical);

/**
 * @brief Map a mouse cursor move to a input function. If you map it to an axis the rate of movement will be applied as rate of change
 *          to the axis. If you map it to a button, the button will be triggered once whenever a rate of change of getAnalogToDigitalRatio
 *          is exceeded in the direction defined by "ab". You can add multiple mappings to an input or use the same cursor change in multiple mappings.
 *          If you map a cursor position change to a button you MUST set "ab" or it will have no effect.
 * @param name Name of the input this mapping should apply to.
 * @param axis Select between horizontal and vertical cursor movement
 * @param requiredMods A bit-set of modifiers that need to be pressed while the cursor is moved for the input to be triggered
 * @param direction The direction in which this needs to be moved in order to trigger a button input.
 */
void mapCourserToInput(std::string name, AxisOrientation axis, int requiredMods = 0, AxisBehavior direction = AxisBehavior::defaultBehavior);

std::string generateHelpString(); //!< generates a string explaining all registered input functions and mappings

}}}
#endif //MPUTILS_INPUT_H
