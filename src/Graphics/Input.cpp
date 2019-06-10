/*
 * mpUtils
 * Imput.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 * Implements the input manager.
 *
 */

// includes
//--------------------
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <mpUtils/Graphics/Input.h>
#include "mpUtils/Log/Log.h"
#include "mpUtils/Graphics/Window.h"
#include <unordered_map>
#include <mpUtils/mpUtils.h>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
namespace Input {
//--------------------

namespace {
    // "global" settings
    std::chrono::milliseconds m_doubleTapTime;
    float m_analogToButtonRatio;
    float m_digitalToAxisRatio;

    // private classes
    /**
     * @brief This class represents a input function. There are different types of input functions (button/axis).
     *          Both can be mapped to all available inputs using different settings for each mapping.
     *          Mapping multiple inputs to one function as well as multiple functions to one input are possible.
     */
    class InputFunction
    {
    public:
        virtual ~InputFunction() =default;
        std::string description; //!< description of the input for display purpose only
        bool active; //!< is the input currently active
        const InputFunctionType type; //!< the type of the input

        virtual void handlePressed(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) {}
        virtual void handleRelesed(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) {}
        virtual void handleRepeat(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) {}
        virtual void handleValue(Window& wnd, double v, AxisBehavior ab) {}

    protected:
        InputFunction(InputFunctionType t, std::string desc, bool isActive)
            : type(t), description(std::move(desc)), active(isActive)
            {}
    };

    /**
     * @brief When a button input is triggered its internal function is called. The window in focus during the event will be passed to that function.
     *        There are different ButtonBehavior that controle when the function is triggered ie onPress or onRelease.
     *        Analog inputs can also be mapped to Buttons. The button is triggered when the analog input surpasses m_analogToButtonRatio.
     */
    class ButtonInput : public InputFunction
    {
    public:
        ButtonInput(std::string desc, bool isActive, ButtonBehavior buttonBehavior, std::function<void(Window&)> func, bool behaviorOverride)
            : InputFunction(InputFunctionType::button, std::move(desc), isActive), defaultBehavior(buttonBehavior),
            allowBehaviorOverride(behaviorOverride), function(std::move(func)), doubleTapTimer(m_doubleTapTime)
        {}

        void handlePressed(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) override
        {
            if(behavior == ButtonBehavior::defaultBehavior || !allowBehaviorOverride)
                behavior = defaultBehavior;

            if(behavior==ButtonBehavior::onPress || behavior==ButtonBehavior::onPressRepeat)
            {
                function(wnd);
            }
            else if(behavior==ButtonBehavior::onDoubleClick)
            {
                doubleTapTimer.update();
                // when the timer is still running, that means we have a double tap
                // if not, restart the timer
                if(doubleTapTimer.isRunning())
                    function(wnd);
                else
                    doubleTapTimer.start();
            }
        }

        void handleRelesed(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) override
        {
            if(behavior == ButtonBehavior::defaultBehavior || !allowBehaviorOverride)
                behavior = defaultBehavior;

            if(behavior==ButtonBehavior::onRelease)
                function(wnd);
        }

        void handleRepeat(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) override
        {
            if(behavior == ButtonBehavior::defaultBehavior || !allowBehaviorOverride)
                behavior = defaultBehavior;

            if(behavior==ButtonBehavior::onRepeat || behavior==ButtonBehavior::onPressRepeat)
                function(wnd);
        }

        void handleValue(Window& wnd, double v, AxisBehavior ab) override
        {
            if(AxisBehavior::positive == ab)
                while( v >= m_analogToButtonRatio)
                {
                    function(wnd);
                    v -= m_analogToButtonRatio;
                }
            else if(AxisBehavior::negative == ab)
                while( v <= -m_analogToButtonRatio)
                {
                    function(wnd);
                    v += m_analogToButtonRatio;
                }
#if !defined(NDEBUG) || defined(MPU_ENABLE_DEBUG_LOGGING)
            else
                logWARNING("InputManager") << "Axis is mapped to button input, but axis behavior is invalid!";
#endif
        }

        mpu::HRTimer doubleTapTimer; //!< timer to be used to time double clicks
        ButtonBehavior defaultBehavior; //!< the buttons default behavior when not overwritten by the input mapping
        bool allowBehaviorOverride; //!< allow that the button behavior can be overwritten by the input mapping
        std::function<void(Window&)> function; //!< function taht performs the actual work
    };

    /**
     * @brief When the value of an analog input changes the input function is called.
     *        The window in focus during the event will be passed to that function, as well as a value that represents
     *        the magnitude and direction of the change. Axis can be bound to digital inputs, in that case a constant
     *        change of m_digitalToAxisRatio is applied in the desired direction while the digital input is pressed.
     */
    class AxisInput : public InputFunction
    {
    public:
        AxisInput(std::string desc, bool isActive, std::function<void(Window&,float)> func)
                : InputFunction(InputFunctionType::button, std::move(desc), isActive), function(std::move(func))
        {}

        void handleValue(Window& wnd, double v, AxisBehavior ab) override
        {
            function(wnd,v);
        }

        std::function<void(Window&,double)> function;
    };

    /**
     * @brief Every instance of this class represents a single mapping between a input device and a input function.
     *          It stores the input function to be called as well as all properties of the mapping, but not the input itself.
     *          That is mostly stored as the key of some map, where an object of type InputMapping is the value.
     */
    class InputMapping
    {
    public:
        InputMapping(std::string name, InputFunction* input, int mods,
                ButtonBehavior behavior, AxisBehavior ab)
                : functionName(std::move(name)), function(input), requiredMods(mods),
                overrideBehavior(behavior), axisBehavior(ab)
        {}

        const std::string functionName; //!< name of the mapped input function
        InputFunction* function; //!< mapped input function
        int requiredMods; //!< required moddifiers to be pressed for this input mapping to be active
        ButtonBehavior overrideBehavior; //!< override button behavior
        AxisBehavior axisBehavior; //!< controles behavior if mapping analog input to button or digital input to axis
    };

    // types
    using InputType = std::pair<std::string,std::unique_ptr<InputFunction>>;
    using InputMapType = std::unordered_map<std::string,std::unique_ptr<InputFunction>>;
    using DigitalMapType = std::unordered_multimap<int, InputMapping>;
    using AnalogMapType = std::vector<InputMapping>;

    // key maps and callback handling
    InputMapType m_inputFunctions; //!< all input funtions life here
    DigitalMapType m_keymap; //!< mapping keyboard keys to input functions
    DigitalMapType m_mbmap; //!< mapping mouse buttons to input functions
    AnalogMapType m_horizontalScrollmap; //!< mapping horizontal scroll movement to input functions
    AnalogMapType m_verticalScrollmap; //!< mapping vertical scroll movement to input functions
    AnalogMapType m_horizontalCursormap; //!< mapping horizontal cursor movement to input functions
    AnalogMapType m_verticalCursormap; //!< mapping vertical cursor movement to input functions

    // data needed for polling and other utilities
    std::vector<Window*> m_wndList; //!< list of all windows that can provide input
    Window* m_focusedWindow; //!< the window in focus or nullptr
    Window* m_hoveredWindow; //!< the window under the cursor or nullptr

    /**
     * @brief initializes the input manager is automatically called when first window is registered
     */
    void initialize()
    {
        m_doubleTapTime = std::chrono::milliseconds(500);
        m_analogToButtonRatio = 1;
        m_digitalToAxisRatio = 1;
    }
}

// declare callback functions
// -----------------------------
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods); //!< key callback of the input manager
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods); //!< mouse button callback of the input manager
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset); //!< scroll callback of the input manager
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos); //!< cursor position callback of the input manager
// -----------------------------

void registerWindow(Window* wnd)
{
    // initialize input manager
    static struct InputInit{
        InputInit()
        {
            initialize();
        }
    }inputInit;

    // add window to lst of windows
    m_wndList.push_back(wnd);

    // register input callbacks with window
    glfwSetKeyCallback(wnd->window(),key_callback);
    glfwSetMouseButtonCallback(wnd->window(),mouse_button_callback);
    glfwSetScrollCallback(wnd->window(),scroll_callback);
    glfwSetCursorPosCallback(wnd->window(),cursor_position_callback);
}

void unregisterWindow(Window* wnd)
{
    auto it = m_wndList.begin();
    while(it != m_wndList.end())
    {
        if(*it == wnd)
            it = m_wndList.erase(it);
        else
            it++;
    }
}

bool isKeyDown(int key)
{
    for(auto &wnd : m_wndList)
    {
        if( wnd->isKeyDown(key) )
            return true;
    }
    return false;
}

bool isMouseButtonDown(int button)
{
    for(auto &wnd : m_wndList)
    {
        if( wnd->isMouseButtonDown(button) )
            return true;
    }
    return false;
}

std::pair<Window*,glm::ivec2> getCursorPos()
{
    for(auto &wnd: m_wndList)
    {
        // TODO: implement with callbacks
    }
}

glm::dvec2 getCursorScreenPos()
{
    glm::dvec2 p(-1,-1);
    if(!m_wndList.empty())
    {
        glfwGetCursorPos(m_wndList[0]->window(), &p.x, &p.y);
        p += m_wndList[0]->getPosition();
    }
    return p;
}

void setCursorScreenPos(double x, double y)
{
    if(!m_wndList.empty())
    {
        auto wpos = m_wndList[0]->getPosition();
        glfwSetCursorPos(m_wndList[0]->window(), x - wpos.x, y - wpos.y);
    }
}

void setCursorScreenPos(glm::dvec2 p)
{
    if(!m_wndList.empty())
    {
        auto wpos = m_wndList[0]->getPosition();
        glfwSetCursorPos(m_wndList[0]->window(), p.x - wpos.x, p.y - wpos.y);
    }
}

Window *getActiveWindow()
{
    // TODO: implement
    return nullptr;
}

Window *getHoveredWindow()
{
    // TODO: implement
    return nullptr;
}

void setCursor(GLFWcursor* c)
{
    for(auto &wnd : m_wndList)
        wnd->setCursor(c);
}

void setCursor(int shape)
{
    for(auto &wnd : m_wndList)
        wnd->setCursor(shape);
}

void setClipboard(const std::string & s)
{
    if(!m_wndList.empty())
        glfwSetClipboardString(m_wndList[0]->window(),s.c_str());
}

std::string getClipboard()
{
    return std::string(glfwGetClipboardString(m_wndList[0]->window()));
}

void setDoubleTapTime(unsigned int ms); //!< sets the time for double taps in miliseconds
unsigned int getDoubleTapTime(); //!< gets the time for double taps in miliseconds

void mapKeyToInput(std::string name, int key, int requiredMods, ButtonBehavior overrideBehavior, AxisBehavior ab)
{
    InputFunction* inputFunc = nullptr;
    auto result = m_inputFunctions.find(name);
    if(result != m_inputFunctions.end())
    {
        inputFunc = result->second.get();
        logDEBUG("InputManager") << "Mapping Key " << key << " to input \"" << name << "\"";
    }
    else
        logWARNING("InputManager") << "Mapping Key " << key << " to input \"" << name << "\" that does not jet exist";

    m_keymap.emplace(key, InputMapping(std::move(name),inputFunc,requiredMods,overrideBehavior, ab));
}

void mapMouseButtonToInput(std::string name, int button, int requiredMods,
                           ButtonBehavior overrideBehavior, AxisBehavior ab)
{
    InputFunction* inputFunc = nullptr;
    auto result = m_inputFunctions.find(name);
    if(result != m_inputFunctions.end())
    {
        inputFunc = result->second.get();
        logDEBUG("InputManager") << "Mapping Key " << button << " to input \"" << name << "\"";
    }
    else
        logWARNING("InputManager") << "Mapping Mouse Button " << button << " to input \"" << name << "\" that does not jet exist";

    m_mbmap.emplace(button, InputMapping(std::move(name),inputFunc,requiredMods,overrideBehavior, ab));
}

void mapScrollToInput(std::string name, int requiredMods, AxisBehavior direction, AxisOrientation axis)
{
    InputFunction* inputFunc = nullptr;
    auto result = m_inputFunctions.find(name);
    if(result != m_inputFunctions.end())
    {
        inputFunc = result->second.get();
        logDEBUG("InputManager") << "Mapping scroll axis " << static_cast<int>(axis) << " to input \"" << name << "\"";
    }
    else
    logWARNING("InputManager") << "Mapping scroll axis " << static_cast<int>(axis) << " to input \"" << name << "\" that does not jet exist";

    if(axis == AxisOrientation::horizontal)
        m_horizontalScrollmap.emplace_back(std::move(name),inputFunc,requiredMods, ButtonBehavior::defaultBehavior, direction);
    else if(axis == AxisOrientation::vertical)
        m_verticalScrollmap.emplace_back(std::move(name),inputFunc,requiredMods, ButtonBehavior::defaultBehavior, direction);
}

void mapCourserToInput(std::string name, AxisOrientation axis, int requiredMods, AxisBehavior direction)
{
    InputFunction* inputFunc = nullptr;
    auto result = m_inputFunctions.find(name);
    if(result != m_inputFunctions.end())
    {
        inputFunc = result->second.get();
        logDEBUG("InputManager") << "Mapping cursor axis " << static_cast<int>(axis) << " to input \"" << name << "\"";
    }
    else
    logWARNING("InputManager") << "Mapping cursor axis " << static_cast<int>(axis) << " to input \"" << name << "\" that does not jet exist";

    if(axis == AxisOrientation::horizontal)
        m_horizontalCursormap.emplace_back(std::move(name),inputFunc,requiredMods, ButtonBehavior::defaultBehavior, direction);
    else if(axis == AxisOrientation::vertical)
        m_verticalCursormap.emplace_back(std::move(name),inputFunc,requiredMods, ButtonBehavior::defaultBehavior, direction);
}

void addButton(std::string name, std::string description, std::function<void(Window&)> function,
        ButtonBehavior behavior, bool allowBehaviorOverride,  bool active)
{
    if(behavior == ButtonBehavior::defaultBehavior)
        behavior = ButtonBehavior::onPress;

    std::unique_ptr<InputFunction> ifunc = std::make_unique<ButtonInput>(std::move(description), active, behavior,
                                                                         std::move(function), allowBehaviorOverride);

    // try to add id
    auto result = m_inputFunctions.insert({std::move(name),std::move(ifunc)});
    if(!result.second)
    {
        logERROR("InputManager") << "Button " << result.first->first
                                 << " could not be added, does already exist with description: "
                                 << result.first->second->description;
        logFlush();
        throw std::logic_error("Input manager button \"" + result.first->first + "\" does already exist and can not be added again.");
    }

    logDEBUG("InputManager") << "Added button \"" << result.first->first << "\"";

    // see if there is already a key map installed thts points to this id
    for(auto &item : m_keymap)
    {
        if(item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing key mapping for button \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_mbmap)
    {
        if(item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing mouse button mapping for button \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_verticalScrollmap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing mouse button mapping for button \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_horizontalScrollmap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing mouse button mapping for button \"" << result.first->first << "\"";
        }
    }
}

void addAxis(std::string name, std::string description, std::function<void(Window&,double)> function, bool active)
{
    std::unique_ptr<InputFunction> ifunc = std::make_unique<AxisInput>(std::move(description), active, std::move(function));

    // try to add id
    auto result = m_inputFunctions.insert({std::move(name),std::move(ifunc)});
    if(!result.second)
    {
        logERROR("InputManager") << "Button " << result.first->first
                                 << " could not be added, does already exist with description: "
                                 << result.first->second->description;
        logFlush();
        throw std::logic_error("Input manager button \"" + result.first->first + "\" does already exist and can not be added again.");
    }

    logDEBUG("InputManager") << "Added button \"" << result.first->first << "\"";

    // see if there is already a key map installed thts points to this id
    for(auto &item : m_keymap)
    {
        if(item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing key mapping for button \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_mbmap)
    {
        if(item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing mouse button mapping for button \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_verticalScrollmap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing mouse button mapping for button \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_horizontalScrollmap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing mouse button mapping for button \"" << result.first->first << "\"";
        }
    }
}

void removeInput(std::string name);
InputFunction* getInput(std::string name);



void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));

    auto range = m_keymap.equal_range(key);
    for(auto it = range.first; it != range.second; ++it)
    {
        // check if the input function is installed
        // check if the input function is active,
        // check if all required modifiers are down,
        if( it->second.function && it->second.function->active && mods >= it->second.requiredMods)
        {
            switch(action)
            {
                case GLFW_PRESS:
                    it->second.function->handlePressed(*wnd,it->second.overrideBehavior, it->second.axisBehavior);
                    break;
                case GLFW_RELEASE:
                    it->second.function->handleRelesed(*wnd,it->second.overrideBehavior, it->second.axisBehavior);
                    break;
                case GLFW_REPEAT:
                    it->second.function->handleRepeat(*wnd,it->second.overrideBehavior, it->second.axisBehavior);
                    break;
                default:
                    break;
            }
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));

    auto range = m_mbmap.equal_range(button);
    for(auto it = range.first; it != range.second; ++it)
    {
        // check if the input function is installed
        // check if the input function is active,
        // check if all required modifiers are down,
        if( it->second.function && it->second.function->active && mods >= it->second.requiredMods)
        {
            switch(action)
            {
                case GLFW_PRESS:
                    it->second.function->handlePressed(*wnd,it->second.overrideBehavior, it->second.axisBehavior);
                    break;
                case GLFW_RELEASE:
                    it->second.function->handleRelesed(*wnd,it->second.overrideBehavior, it->second.axisBehavior);
                    break;
                case GLFW_REPEAT:
                    it->second.function->handleRepeat(*wnd,it->second.overrideBehavior, it->second.axisBehavior);
                    break;
                default:
                    break;
            }
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));

    int mods = 0;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_SHIFT) || wnd->isKeyDown(GLFW_KEY_RIGHT_SHIFT))
        mods |= GLFW_MOD_SHIFT;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_CONTROL) || wnd->isKeyDown(GLFW_KEY_RIGHT_CONTROL))
        mods |= GLFW_MOD_CONTROL;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_ALT) || wnd->isKeyDown(GLFW_KEY_RIGHT_ALT))
        mods |= GLFW_MOD_ALT;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_SUPER) || wnd->isKeyDown(GLFW_KEY_RIGHT_SUPER))
        mods |= GLFW_MOD_SUPER;

    for(auto &item : m_verticalScrollmap)
    {
        // check if the input function is installed
        // check if the input function is active,
        // check if all required modifiers are down,
        if( item.function && item.function->active && mods >= item.requiredMods)
        {
            item.function->handleValue(*wnd, yoffset, item.axisBehavior);
        }
    }

    for(auto &item : m_horizontalScrollmap)
    {
        // check if the input function is installed
        // check if the input function is active,
        // check if all required modifiers are down,
        if( item.function && item.function->active && mods >= item.requiredMods)
        {
            item.function->handleValue(*wnd, xoffset, item.axisBehavior);
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));

    int mods = 0;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_SHIFT) || wnd->isKeyDown(GLFW_KEY_RIGHT_SHIFT))
        mods |= GLFW_MOD_SHIFT;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_CONTROL) || wnd->isKeyDown(GLFW_KEY_RIGHT_CONTROL))
        mods |= GLFW_MOD_CONTROL;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_ALT) || wnd->isKeyDown(GLFW_KEY_RIGHT_ALT))
        mods |= GLFW_MOD_ALT;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_SUPER) || wnd->isKeyDown(GLFW_KEY_RIGHT_SUPER))
        mods |= GLFW_MOD_SUPER;

    // calculate the rate of change
    static double xLastPos=xpos;
    static double yLastPos=ypos;
    double xChange = xpos - xLastPos;
    double yChange = ypos - yLastPos;
    xLastPos = xpos;
    yLastPos = ypos;

    if( std::fabs(yChange) > 0)
        for(auto &item : m_verticalCursormap)
            {
                // check if the input function is installed
                // check if the input function is active,
                // check if all required modifiers are down,
                if( item.function && item.function->active && mods >= item.requiredMods)
                {
                    item.function->handleValue(*wnd, yChange, item.axisBehavior);
                }
            }

    if( std::fabs(xChange) > 0)
        for(auto &item : m_horizontalCursormap)
        {
            // check if the input function is installed
            // check if the input function is active,
            // check if all required modifiers are down,
            if( item.function && item.function->active && mods >= item.requiredMods)
            {
                item.function->handleValue(*wnd, xChange, item.axisBehavior);
            }
        }
}

}}}