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
    std::chrono::milliseconds m_doubleClickTime;
    double m_analogToButtonRatio;
    double m_digitalToAxisRatio;
    double m_mouseSensitivityX;
    double m_mouseSensitivityY;
    double m_scrollSensitivityX;
    double m_scrollSensitivityY;
    bool m_mouseInputEnabled = true;
    bool m_cursorInputEnabled = true;
    bool m_keyboardInputEnabled = true;

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
        virtual void handleIsDown(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) {} //!< called during polling

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
        ButtonInput(std::string desc, bool isActive, std::function<void(Window&)> func)
            : InputFunction(InputFunctionType::button, std::move(desc), isActive), function(std::move(func)), doubleTapTimer(m_doubleClickTime)
        {}

        void handlePressed(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) override
        {
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
            if(behavior==ButtonBehavior::onRelease)
                function(wnd);
        }

        void handleRepeat(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) override
        {
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

        void handleIsDown(Window &wnd, ButtonBehavior behavior, AxisBehavior ab) override
        {
            if(behavior == ButtonBehavior::whenDown)
                function(wnd);
        }

        mpu::HRTimer doubleTapTimer; //!< timer to be used to time double clicks
        std::function<void(Window&)> function; //!< function that performs the actual work
    };

    /**
     * @brief When the value of an analog input changes the input function is called.
     *        The window in focus during the event will be passed to that function, as well as a value that represents
     *        the magnitude and direction of the change. Axis can be bound to digital inputs, in that case a
     *        change of m_digitalToAxisRatio per second is applied in the desired direction while the digital input is pressed.
     */
    class AxisInput : public InputFunction
    {
    public:
        AxisInput(std::string desc, bool isActive, std::function<void(Window&,float)> func)
                : InputFunction(InputFunctionType::axis, std::move(desc), isActive), function(std::move(func))
        {}

        void handleValue(Window& wnd, double v, AxisBehavior ab) override
        {
            function(wnd, static_cast<float>(ab)*v);
        }

        void handleIsDown(Window &wnd, ButtonBehavior behavior, AxisBehavior ab) override
        {
            function(wnd, static_cast<float>(ab)*m_digitalToAxisRatio*deltaTime());
        }

        std::function<void(Window&,double)> function;
    };

    class CustomModifier : public InputFunction
    {
    public:
        explicit CustomModifier(std::string desc)
                : InputFunction(InputFunctionType::customModifier, std::move(desc), true), m_isDown(false)
        {
        }

        void handlePressed(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) override
        {
            m_isDown = true;
        }

        void handleRelesed(Window& wnd, ButtonBehavior behavior, AxisBehavior ab) override
        {
            m_isDown = false;
        }

        bool isDown() {return m_isDown;}

    private:
        bool m_isDown;
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
                     ButtonBehavior behavior, AxisBehavior ab, CustomModifier* customModifier)
                : functionName(std::move(name)), function(input), requiredMods(mods),
                  buttonBehavior(behavior), axisBehavior(ab), customMod(customModifier)
        {}

        /**
         * Returns true if all conditions are met for the mapping to be triggered
         * @param mods the currently activated modifier keys
         * @return true if the mapping can be activated
         */
        bool readyToUse(int mods)
        {
            return function && function->active && (((requiredMods & mods) ^requiredMods) == 0) && (customMod == nullptr || customMod->isDown());
        }

        const std::string functionName; //!< name of the mapped input function
        InputFunction* function; //!< mapped input function
        int requiredMods; //!< required modifiers to be pressed for this input mapping to be active
        CustomModifier* customMod; //!< points to the custom modifier added to this mapping
        ButtonBehavior buttonBehavior; //!< the button behavior when mapped to a button input function
        AxisBehavior axisBehavior; //!< controls behavior if mapping analog input to button or digital input to axis
    };

    // key maps and InputFunction handling
    using InputType = std::pair<std::string,std::unique_ptr<InputFunction>>;
    using InputMapType = std::unordered_map<std::string,std::unique_ptr<InputFunction>>;
    using DigitalMapType = std::unordered_multimap<int, InputMapping>;
    using PollListType = std::vector<std::pair<int, InputMapping>>;
    using AnalogMapType = std::vector<InputMapping>;

    InputMapType m_inputFunctions; //!< all input funtions life here
    DigitalMapType m_keymap; //!< mapping keyboard keys to input functions
    DigitalMapType m_mbmap; //!< mapping mouse buttons to input functions
    AnalogMapType m_horizontalScrollmap; //!< mapping horizontal scroll movement to input functions
    AnalogMapType m_verticalScrollmap; //!< mapping vertical scroll movement to input functions
    AnalogMapType m_horizontalCursormap; //!< mapping horizontal cursor movement to input functions
    AnalogMapType m_verticalCursormap; //!< mapping vertical cursor movement to input functions
    PollListType m_polledKeys; //!< all keyboard keys that need to be polled because for some reason
    PollListType m_polledMbs; //!< all mouse buttons that need to be polled because for some reason

    // lists to store external callbacks
    std::vector<std::pair<int,std::function<void(Window&,const std::vector<std::string>&)>>> m_dropCallbacks;
    std::vector<std::pair<int,std::function<void(Window&,bool)>>> m_cursorEnterCallbacks;
    std::vector<std::pair<int,std::function<void(Window&,unsigned int)>>> m_charCallbacks;
    std::vector<std::pair<int,std::function<void()>>> m_updateCallbacks;

    // data needed for polling and other utilities
    std::vector<Window*> m_wndList; //!< list of all windows that can provide input
    Window* m_focusedWindow; //!< the window in focus or nullptr
    Window* m_hoveredWindow; //!< the window under the cursor or nullptr

    double m_lastTime; //!< the time returned by the last call to glfwGetTime
    double m_frametime; //!< the time since the last call to Input::update();

    // private helper functions

    void initialize(); //!< initializes the input manager is automatically called when first window is registered
    template<typename T, typename F> int addCallback(std::vector<std::pair<int,T>> &callbackVector, F f); //!< internal helper to add a callback function to vector of callbacks
    template<typename T> void removeCallback(std::vector<std::pair<int,T>> &callbackVector, int id); //!< internal helper to remove a callback function from vector of callbacks
    CustomModifier* addCustomModifier(std::string name, std::string description); //!< Creates a Input of type custom modifier and adds it to the list of inputs.

    // declare glfw callback functions
    void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods); //!< key callback of the input manager
    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods); //!< mouse button callback of the input manager
    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset); //!< scroll callback of the input manager
    void cursor_position_callback(GLFWwindow* window, double xpos, double ypos); //!< cursor position callback of the input manager
    void drop_callback(GLFWwindow* window, int count, const char** paths); //!< drop callback of the input manager
    void cursor_enter_callback(GLFWwindow* window, int entered); //!< cursor enter callback of the input manager
    void character_callback(GLFWwindow* window, unsigned int codepoint); // char callback of the input manager
}

// functions only used by other mpUtils Modules

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
    glfwSetDropCallback(wnd->window(),drop_callback);
    glfwSetCursorEnterCallback(wnd->window(), cursor_enter_callback);
    glfwSetCharCallback(wnd->window(), character_callback);

    // register focus callback to keep track of the window in focus
    wnd->addFocusCallback([wnd](bool f)
    {
        if(f)
            m_focusedWindow = wnd;
        else if(wnd == m_focusedWindow)
            m_focusedWindow = nullptr;
    });
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

// update function to update the imput state and handle all callbacks
void update()
{
    // update time
    double t = glfwGetTime();
    m_frametime = t - m_lastTime;
    m_lastTime = t;

    // handle glfw
    glfwPollEvents();

    // now handle input mappings that need to be polled
    for(auto &wnd : m_wndList)
    {
        int mods = 0;
        if(wnd->isKeyDown(GLFW_KEY_LEFT_SHIFT) || wnd->isKeyDown(GLFW_KEY_RIGHT_SHIFT))
            mods |= GLFW_MOD_SHIFT;
        if(wnd->isKeyDown(GLFW_KEY_LEFT_CONTROL) || wnd->isKeyDown(GLFW_KEY_RIGHT_CONTROL))
            mods |= GLFW_MOD_CONTROL;
        if(wnd->isKeyDown(GLFW_KEY_LEFT_ALT) || wnd->isKeyDown(GLFW_KEY_RIGHT_ALT))
            mods |= GLFW_MOD_ALT;
        if(wnd->isKeyDown(GLFW_KEY_LEFT_SUPER) || wnd->isKeyDown(GLFW_KEY_RIGHT_SUPER))
            mods |= GLFW_MOD_SUPER;

        if(m_keyboardInputEnabled)
            for(auto &item : m_polledKeys)
            {
                if( wnd->isKeyDown(item.first) && item.second.readyToUse(mods))
                {
                    item.second.function->handleIsDown(*wnd,item.second.buttonBehavior,item.second.axisBehavior);
                }
            }

        if(m_mouseInputEnabled)
            for(auto &item : m_polledMbs)
            {
                if( wnd->isMouseButtonDown(item.first) && item.second.readyToUse(mods))
                {
                    item.second.function->handleIsDown(*wnd,item.second.buttonBehavior,item.second.axisBehavior);
                }
            }
    }

    //call all update callbacks
    for(auto &callback : m_updateCallbacks)
    {
        callback.second();
    }
}

void disableMouseInput()
{
    m_mouseInputEnabled = false;
}

void enableMouseInput()
{
    m_mouseInputEnabled = true;
}

bool isMouseInputEnabled()
{
    return m_mouseInputEnabled;
}

void toggleMouseInput()
{
    if(isMouseInputEnabled())
        disableMouseInput();
    else
        enableMouseInput();
}

void disableCursourInput()
{
    m_cursorInputEnabled = false;
}

void enableCursourInput()
{
    m_cursorInputEnabled = true;
}

bool isCursourInputEnabled()
{
    return m_cursorInputEnabled;
}

void toggleCursourInput()
{
    if(isCursourInputEnabled())
        disableCursourInput();
    else
        enableCursourInput();
}

void disableKeyboardInput()
{
    m_keyboardInputEnabled = false;
}

void enableKeyboardInput()
{
    m_keyboardInputEnabled = true;
}

bool isKeyboardInputEnabled()
{
    return m_keyboardInputEnabled;
}

void toggleKeyboardInput()
{
    if(isKeyboardInputEnabled())
        disableKeyboardInput();
    else
        enableKeyboardInput();
}

double deltaTime()
{
    return m_frametime;
}

// functions for input polling

bool isKeyDown(int key)
{
    if(!m_keyboardInputEnabled)
        return false;

    for(auto &wnd : m_wndList)
    {
        if( wnd->isKeyDown(key) )
            return true;
    }
    return false;
}

bool isMouseButtonDown(int button)
{
    if(!m_mouseInputEnabled)
        return false;

    for(auto &wnd : m_wndList)
    {
        if( wnd->isMouseButtonDown(button) )
            return true;
    }
    return false;
}

std::pair<Window*,glm::dvec2> getCursorPos()
{
    Window* wnd = getHoveredWindow();
    if(!wnd)
        wnd = m_wndList[0];
    return {wnd,wnd->getCursorPos()};
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

Window *getActiveWindow()
{
    return m_focusedWindow;
}

Window *getHoveredWindow()
{
    return m_hoveredWindow;
}

// functions to control the cursor

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

// handle clippboard

void setClipboard(const std::string & s)
{
    if(!m_wndList.empty())
        glfwSetClipboardString(m_wndList[0]->window(),s.c_str());
}

std::string getClipboard()
{
    return std::string(glfwGetClipboardString(m_wndList[0]->window()));
}

// remove and add custom callbacks

int addUpdateCallback(std::function<void()> f)
{
    return addCallback(m_updateCallbacks, std::move(f));
}

void removeUpdateCallback(int id)
{
    removeCallback(m_updateCallbacks, id);
}

int addDropCallback(std::function<void(Window&, const std::vector<std::string>&)> f)
{
    return addCallback(m_dropCallbacks, std::move(f));
}

void removeDropCallback(int id)
{
    removeCallback(m_dropCallbacks, id);
}

int addCursorEnterCallback(std::function<void(Window &, bool)> f)
{
    return addCallback(m_cursorEnterCallbacks, std::move(f));
}

void removeCursorEnterCallback(int id)
{
    removeCallback(m_cursorEnterCallbacks, id);
}

int addCharCallback(std::function<void(Window &, unsigned int)> f)
{
    return addCallback(m_charCallbacks, std::move(f));
}

void removeCharCallback(int id)
{
    removeCallback(m_charCallbacks, id);
}

// changing global settings

void setDoubleClickTime(unsigned int ms)
{
    m_doubleClickTime = std::chrono::milliseconds(ms);
}

unsigned int getDoubleClickTime()
{
    return static_cast<unsigned int>(m_doubleClickTime.count());
}

void setAnalogToDigitalRatio(double r)
{
    m_analogToButtonRatio = r;
}

double getAnalogToDigitalRatio()
{
    return m_analogToButtonRatio;
}

void setDigitaltoAnalogRatio(double r)
{
    m_digitalToAxisRatio = r;
}

double getDigitalToAnalogRatio()
{
    return m_digitalToAxisRatio;
}

void setMouseSensitivityX(double sX)
{
    m_mouseSensitivityX = sX;
}

void setMouseSensitivityY(double sY)
{
    m_mouseSensitivityY = sY;
}

void setScrollSensitivityX(double sX)
{
    m_scrollSensitivityX = sX;
}

void setScrollSensitivityY(double sY)
{
    m_scrollSensitivityY = sY;
}

double getMouseSensitivityX()
{
    return m_mouseSensitivityX;
}

double getMouseSensitivityY()
{
    return m_mouseSensitivityY;
}

double getScrollSensitivityX()
{
    return m_scrollSensitivityX;
}

double getScrollSensitivityY()
{
    return m_scrollSensitivityY;
}

// managed input handling

void mapKeyToInput(std::string name, int key, ButtonBehavior buttonBehavior, AxisBehavior ab, int requiredMods, std::string customModifierName)
{
    InputFunction* inputFunc = nullptr;
    auto result = m_inputFunctions.find(name);
    if(result != m_inputFunctions.end())
    {
        inputFunc = result->second.get();
        logDEBUG("InputManager") << "Mapping Key "<< key << " (\"" << ((glfwGetKeyName(key,0))? glfwGetKeyName(key,0) : " ") << "\") to input \"" << name << "\"";
    }
    else
        logWARNING("InputManager") << "Mapping Key "<< key << " (\"" << ((glfwGetKeyName(key,0))? glfwGetKeyName(key,0) : " ") << "\") to input \"" << name << "\" that does not jet exist";

    // create a custom mod if needed
    CustomModifier* cmod = nullptr;
    if(!customModifierName.empty())
        cmod = addCustomModifier(std::move(customModifierName), "Activates the key " + std::string(((glfwGetKeyName(key,0))? glfwGetKeyName(key,0) : " ")) + " mapped to " + name + ".");

    // check if this needs to be polled for some reason
    if(    inputFunc && (  ( inputFunc->type == InputFunctionType::axis) // the input is an axis => needs to be polled
                           || ( inputFunc->type == InputFunctionType::button
                                && buttonBehavior == ButtonBehavior::whenDown // button behavior is whenDown
                           )
                        ))
    {
        m_polledKeys.emplace_back(key, InputMapping(std::move(name), inputFunc, requiredMods, buttonBehavior, ab, cmod));
    }
    else
    {
        m_keymap.emplace(key, InputMapping(std::move(name), inputFunc, requiredMods, buttonBehavior, ab, cmod));
    }
}

void mapMouseButtonToInput(std::string name, int button, ButtonBehavior buttonBehavior,
                           AxisBehavior ab, int requiredMods, std::string customModifierName)
{
    InputFunction* inputFunc = nullptr;
    auto result = m_inputFunctions.find(name);
    if(result != m_inputFunctions.end())
    {
        inputFunc = result->second.get();
        logDEBUG("InputManager") << "Mapping mouse button " << button << " to input \"" << name << "\"";
    }
    else
        logWARNING("InputManager") << "Mapping mouse button " << button << " to input \"" << name << "\" that does not jet exist";

    // create a custom mod if needed
    CustomModifier* cmod = nullptr;
    if(!customModifierName.empty())
    {
        cmod = addCustomModifier(std::move(customModifierName), "Activates the mouse button " + std::to_string(button) + " mapped to " + name + ".");
    }

    // check if this needs to be polled for some reason
    if(    inputFunc && (  ( inputFunc->type == InputFunctionType::axis) // the input is an axis => needs to be polled
                           || ( inputFunc->type == InputFunctionType::button
                                && buttonBehavior == ButtonBehavior::whenDown // button behavior is whenDown
                           )
    ))
    {
        m_polledMbs.emplace_back(button, InputMapping(std::move(name), inputFunc, requiredMods, buttonBehavior, ab, cmod));
    }
    else
    {
        m_mbmap.emplace(button, InputMapping(std::move(name), inputFunc, requiredMods, buttonBehavior, ab, cmod));
    }
}

void mapScrollToInput(std::string name, AxisBehavior direction, int requiredMods, std::string customModifierName, AxisOrientation axis)
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

    // create a custom mod if needed
    CustomModifier* cmod = nullptr;
    if(!customModifierName.empty())
    {
        cmod = addCustomModifier(std::move(customModifierName), "Activates the scroll axis " + std::to_string(static_cast<int>(axis)) + " mapped to " + name + ".");
    }

    if(axis == AxisOrientation::horizontal)
        m_horizontalScrollmap.emplace_back(std::move(name),inputFunc,requiredMods, ButtonBehavior::other, direction, cmod);
    else if(axis == AxisOrientation::vertical)
        m_verticalScrollmap.emplace_back(std::move(name),inputFunc,requiredMods, ButtonBehavior::other, direction, cmod);
}

void mapCourserToInput(std::string name, AxisOrientation axis, AxisBehavior direction, int requiredMods, std::string customModifierName)
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

    // create a custom mod if needed
    CustomModifier* cmod = nullptr;
    if(!customModifierName.empty())
    {
        cmod = addCustomModifier(std::move(customModifierName), "Activates the cursor axis " + std::to_string(static_cast<int>(axis)) + " mapped to " + name + ".");
    }

    if(axis == AxisOrientation::horizontal)
        m_horizontalCursormap.emplace_back(std::move(name),inputFunc,requiredMods, ButtonBehavior::other, direction, cmod);
    else if(axis == AxisOrientation::vertical)
        m_verticalCursormap.emplace_back(std::move(name),inputFunc,requiredMods, ButtonBehavior::other, direction, cmod);
}

void addButton(std::string name, std::string description, std::function<void(Window&)> function, bool active)
{
    std::unique_ptr<InputFunction> ifunc = std::make_unique<ButtonInput>(std::move(description), active, std::move(function));

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

    // see if there is already a key map installed that points to this id

    auto iter = m_keymap.begin();
    while( iter != m_keymap.end())
    {
        auto buffer = iter;
        auto& item = *iter;
        if(item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing key mapping for button \"" << result.first->first << "\"";
            // check if we need to poll this
            if( item.second.buttonBehavior == ButtonBehavior::whenDown)
            {
                m_polledKeys.emplace_back(item.first, item.second);
                iter = m_keymap.erase(iter);
            }
        }
        if(buffer==iter)
            iter++;
    }

    auto it = m_mbmap.begin();
    while( it != m_mbmap.end())
    {
        auto buffer = it;
        auto& item = *it;
        if(item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing mouse button mapping for button \"" << result.first->first << "\"";
            // check if we need to poll this
            if( item.second.buttonBehavior == ButtonBehavior::whenDown)
            {
                m_polledMbs.emplace_back(item.first, item.second);
                it = m_mbmap.erase(it);
            }
        }
        if(buffer==it)
            it++;
    }

    for(auto &item : m_verticalScrollmap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing scroll mapping for button \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_horizontalScrollmap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing scroll mapping for button \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_verticalCursormap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing cursor mapping for button \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_horizontalCursormap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing cursor mapping for button \"" << result.first->first << "\"";
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

    logDEBUG("InputManager") << "Added Axis \"" << result.first->first << "\"";

    // see if there is already a key map installed thts points to this id
    auto iter = m_keymap.begin();
    while( iter != m_keymap.end())
    {
        auto& item = *iter;
        if(item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing key mapping for axis \"" << result.first->first << "\"";
            // check if we need to poll this
            m_polledKeys.emplace_back(item.first, item.second);
            iter = m_keymap.erase(iter);
        }
        else
            iter++;
    }

    auto it = m_mbmap.begin();
    while( it != m_mbmap.end())
    {
        auto& item = *it;
        if(item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing mouse button mapping for axis \"" << result.first->first << "\"";
            // check if we need to poll this
            m_polledMbs.emplace_back(item.first, item.second);
            it = m_mbmap.erase(it);
        }
        else
            it++;
    }

    for(auto &item : m_verticalScrollmap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing scroll mapping for axis \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_horizontalScrollmap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing scroll mapping for axis \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_verticalCursormap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing cursor mapping for axis \"" << result.first->first << "\"";
        }
    }

    for(auto &item : m_horizontalCursormap)
    {
        if(item.functionName == result.first->first)
        {
            item.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing cursor mapping for axis \"" << result.first->first << "\"";
        }
    }
}


namespace {

// private functions

void initialize()
{
    m_hoveredWindow = nullptr;
    m_focusedWindow = nullptr;
    m_doubleClickTime = std::chrono::milliseconds(500);
    m_analogToButtonRatio = 1;
    m_digitalToAxisRatio = 20;
    m_frametime = 0;
    m_lastTime = glfwGetTime();
    m_mouseSensitivityX = 1;
    m_mouseSensitivityY = 1;
    m_scrollSensitivityX = 1;
    m_scrollSensitivityY = 1;
}

template<typename T, typename F>
int addCallback(std::vector<std::pair<int,T>> &callbackVector, F f)
{
    int id;
    if(callbackVector.empty())
        id = 0;
    else
        id = callbackVector.back().first+1;

    callbackVector.emplace_back(id, f);
    return id;
}

template<typename T>
void removeCallback(std::vector<std::pair<int,T>> &callbackVector, int id)
{
    auto it = std::lower_bound( callbackVector.cbegin(), callbackVector.cend(), std::pair<int,T>(id,T{}),
                                [](const std::pair<int,T>& a, const std::pair<int,T>& b){return (a.first < b.first);});
    if(it != callbackVector.end())
        callbackVector.erase(it);
}

CustomModifier *addCustomModifier(std::string name, std::string description)
{
    std::unique_ptr<InputFunction> ifunc = std::make_unique<CustomModifier>(std::move(description));
    auto ptr = dynamic_cast<CustomModifier *>(ifunc.get());

    // try to add id
    auto result = m_inputFunctions.insert({std::move(name), std::move(ifunc)});
    if (!result.second)
    {
        logWARNING("InputManager") << "CustomModifier " << result.first->first
                                   << " could not be added, existing modifier will be used, with description: "
                                   << result.first->second->description;
        InputFunction* f = result.first->second.get();
        if(f && f->type == InputFunctionType::customModifier)
        {
            return dynamic_cast<CustomModifier*>(f);
        } else
        {
            logERROR("InputManager") << "CustomModifier " << result.first->first
                                     << " could not be added, another input function of different type with the same name exists.";
            logFlush();
            throw std::logic_error("Custom Modifer could not be added, another input function of different type exists with the same name.");
        }
    }

    logDEBUG("InputManager") << "Added CustomModifier \"" << result.first->first << "\"";

    // see if there is already a key map installed that points to this id

    auto iter = m_keymap.begin();
    while (iter != m_keymap.end())
    {
        auto &item = *iter;
        if (item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing key mapping for CustomModifier \"" << result.first->first << "\"";
        }
        iter++;
    }

    auto it = m_mbmap.begin();
    while (it != m_mbmap.end())
    {
        auto &item = *it;
        if (item.second.functionName == result.first->first)
        {
            item.second.function = result.first->second.get();
            logDEBUG("InputManager") << "Found existing mouse button mapping for CustomModifier \"" << result.first->first
                                     << "\"";
        }
        it++;
    }

    return ptr;
}

// internal callbacks

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(!m_keyboardInputEnabled)
        return;

    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));

    auto range = m_keymap.equal_range(key);
    for(auto it = range.first; it != range.second; ++it)
    {
        // check if the input function is installed
        // check if the input function is active,
        // check if all required modifiers are down,
        if( it->second.readyToUse(mods))
        {
            switch(action)
            {
                case GLFW_PRESS:
                    it->second.function->handlePressed(*wnd,it->second.buttonBehavior, it->second.axisBehavior);
                    break;
                case GLFW_RELEASE:
                    it->second.function->handleRelesed(*wnd,it->second.buttonBehavior, it->second.axisBehavior);
                    break;
                case GLFW_REPEAT:
                    it->second.function->handleRepeat(*wnd,it->second.buttonBehavior, it->second.axisBehavior);
                    break;
                default:
                    break;
            }
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if(!m_mouseInputEnabled)
        return;

    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));

    auto range = m_mbmap.equal_range(button);
    for(auto it = range.first; it != range.second; ++it)
    {
        // check if the input function is installed
        // check if the input function is active,
        // check if all required modifiers are down,
        if( it->second.readyToUse(mods))
        {
            switch(action)
            {
                case GLFW_PRESS:
                    it->second.function->handlePressed(*wnd,it->second.buttonBehavior, it->second.axisBehavior);
                    break;
                case GLFW_RELEASE:
                    it->second.function->handleRelesed(*wnd,it->second.buttonBehavior, it->second.axisBehavior);
                    break;
                case GLFW_REPEAT:
                    it->second.function->handleRepeat(*wnd,it->second.buttonBehavior, it->second.axisBehavior);
                    break;
                default:
                    break;
            }
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if(!m_mouseInputEnabled)
        return;

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

    // apply sensitivity
    xoffset *= m_scrollSensitivityX;
    yoffset *= m_scrollSensitivityY;

    for(auto &item : m_verticalScrollmap)
    {
        // check if the input function is installed
        // check if the input function is active,
        // check if all required modifiers are down,
        if( item.readyToUse(mods))
        {
            item.function->handleValue(*wnd, yoffset, item.axisBehavior);
        }
    }

    for(auto &item : m_horizontalScrollmap)
    {
        // check if the input function is installed
        // check if the input function is active,
        // check if all required modifiers are down,
        if( item.readyToUse(mods))
        {
            item.function->handleValue(*wnd, xoffset, item.axisBehavior);
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if(!m_cursorInputEnabled)
        return;

    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));

    int mods = 0;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_SHIFT) || wnd->isKeyDown(GLFW_KEY_RIGHT_SHIFT))
        mods |= GLFW_MOD_SHIFT;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_CONTROL)  || wnd->isKeyDown(GLFW_KEY_RIGHT_CONTROL))
        mods |= GLFW_MOD_CONTROL;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_ALT)  || wnd->isKeyDown(GLFW_KEY_RIGHT_ALT))
        mods |= GLFW_MOD_ALT;
    if(wnd->isKeyDown(GLFW_KEY_LEFT_SUPER)  || wnd->isKeyDown(GLFW_KEY_RIGHT_SUPER))
        mods |= GLFW_MOD_SUPER;

    // calculate the rate of change
    static double xLastPos=xpos;
    static double yLastPos=ypos;
    double xChange = xpos - xLastPos;
    double yChange = ypos - yLastPos;
    xLastPos = xpos;
    yLastPos = ypos;

    // apply sensitivity
    xChange *= m_mouseSensitivityX;
    yChange *= m_mouseSensitivityY;

    if( std::fabs(yChange) > 0)
        for(auto &item : m_verticalCursormap)
        {
            // check if the input function is installed
            // check if the input function is active,
            // check if all required modifiers are down,
            if( item.readyToUse(mods))
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
            if( item.readyToUse(mods))
            {
                item.function->handleValue(*wnd, xChange, item.axisBehavior);
            }
        }
}

void drop_callback(GLFWwindow* window, int count, const char** paths)
{
    logDEBUG("InputManager") << "Drop event recieved. " << count << " files dropped.";

    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));
    std::vector<std::string> data(count);
    for(int i = 0; i < count; ++i)
    {
        data[i] = paths[i];
    }

    for(auto &callback : m_dropCallbacks)
    {
        callback.second(*wnd,data);
    }
}

void cursor_enter_callback(GLFWwindow* window, int entered)
{
    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (entered)
    {
        m_hoveredWindow = wnd;
    }
    else if(m_hoveredWindow == wnd)
    {
        // cursor left the hovered window
        m_hoveredWindow = nullptr;
    }

    for(auto &callback : m_cursorEnterCallbacks)
    {
        callback.second(*wnd,entered);
    }
}

void character_callback(GLFWwindow* window, unsigned int codepoint)
{
    if(!m_keyboardInputEnabled)
        return;

    auto wnd = static_cast<Window*>(glfwGetWindowUserPointer(window));
    for(auto &callback : m_charCallbacks)
    {
        callback.second(*wnd,codepoint);
    }
}


}

}}}