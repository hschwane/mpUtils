/*
 * mpUtils
 * Imput.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
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
    std::vector<Window*> m_wndList; //!< list of all windows that can provide input
    Window* m_focusedWindow; //!< the window in focus or nullptr
    Window* m_hoveredWindow; //!< the window under the cursor or nullptr


    std::chrono::milliseconds m_doubleTapTime(500);
    float m_axisToButtonRatio = 1;



    struct InputFunction
    {
        virtual ~InputFunction() =default;
        std::string description; //!< description of the input for display purpose only
        bool active; //!< is the input currently active
        const InputFunctionType type; //!< the type of the input

        virtual void handlePressed(Window& wnd, ButtonBehavior behavior) {}
        virtual void handleRelesed(Window& wnd, ButtonBehavior behavior) {}
        virtual void handleRepeat(Window& wnd, ButtonBehavior behavior) {}
        virtual void handleValue(Window& wnd, float v) {}

    protected:
        InputFunction(InputFunctionType t, std::string desc, bool isActive)
            : type(t), description(std::move(desc)), active(isActive)
            {}
    };

    struct ButtonInput : public InputFunction
    {
        ButtonInput(std::string desc, bool isActive, ButtonBehavior buttonBehavior, std::function<void(Window&)> func, bool behaviorOverride)
            : InputFunction(InputFunctionType::button, std::move(desc), isActive), defaultBehavior(buttonBehavior),
            allowBehaviorOverride(behaviorOverride), function(std::move(func)), doubleTapTimer(m_doubleTapTime)
        {}

        void handlePressed(Window& wnd, ButtonBehavior behavior) override
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

        void handleRelesed(Window& wnd, ButtonBehavior behavior) override
        {
            if(behavior == ButtonBehavior::defaultBehavior || !allowBehaviorOverride)
                behavior = defaultBehavior;

            if(behavior==ButtonBehavior::onRelease)
                function(wnd);
        }

        void handleRepeat(Window& wnd, ButtonBehavior behavior) override
        {
            if(behavior == ButtonBehavior::defaultBehavior || !allowBehaviorOverride)
                behavior = defaultBehavior;

            if(behavior==ButtonBehavior::onRepeat || behavior==ButtonBehavior::onPressRepeat)
                function(wnd);
        }

        void handleValue(Window& wnd, float v) override
        {
            while( v > m_axisToButtonRatio)
            {
                function(wnd);
                v -= m_axisToButtonRatio;
            }
        }

        mpu::HRTimer doubleTapTimer; //!< timer to be used to time double clicks
        ButtonBehavior defaultBehavior; //!< the buttons default behavior when not overwritten by the input mapping
        bool allowBehaviorOverride; //!< allow that the button behavior can be overwritten by the input mapping
        std::function<void(Window&)> function; //!< function taht performs the actual work
    };

    struct AxisInput : public InputFunction
    {
        std::function<void(Window&,float)> onChange;
    };

    struct InputMapping
    {
        InputMapping(std::string name, InputFunction* input, int mods = 0,
                ButtonBehavior behavior = ButtonBehavior::defaultBehavior)
                : functionName(std::move(name)), function(input), requiredMods(mods), overrideBehavior(behavior)
        {}

        const std::string functionName; //!< name of the mapped input function
        InputFunction* function; //!< mapped input function
        int requiredMods; //!< required moddifiers to be pressed for this input mapping to be active
        ButtonBehavior overrideBehavior; //!< override button behavior
    };

    using InputType = std::pair<std::string,std::unique_ptr<InputFunction>>;
    using InputMapType = std::unordered_map<std::string,std::unique_ptr<InputFunction>>;
    using KeymapType = std::unordered_multimap<int, InputMapping>;

    InputMapType m_inputFunctions;
    KeymapType m_keymap;
}

// declare callback functions
// -----------------------------
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods); //!< key callback of the input manager
// -----------------------------

void registerWindow(Window* wnd)
{
    m_wndList.push_back(wnd);
    glfwSetKeyCallback(wnd->window(),key_callback);
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

void bindKeyToInput(std::string name, int key, int requiredMods, ButtonBehavior overrideBehavior)
{
    InputFunction* inputFunc = nullptr;
    auto result = m_inputFunctions.find(name);
    if(result != m_inputFunctions.end())
        inputFunc = result->second.get();
    else
        logWARNING("InputManager") << "Mapping Key " << key << " to input \"" << name << "\" that does not jet exist";

    m_keymap.emplace(key, InputMapping(std::move(name),inputFunc,requiredMods,overrideBehavior));
}

void bindMouseButtonToInput(std::string name);
void bindCourserToInput(std::string name);
void bindScrollToInput(std::string name);

void addButton(std::string name, std::string description, std::function<void(Window&)> function,
        ButtonBehavior behavior, bool allowBehaviorOverride, bool active)
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
}

void addAxis(std::string name, std::string description);

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
                    it->second.function->handlePressed(*wnd,it->second.overrideBehavior);
                    break;
                case GLFW_RELEASE:
                    it->second.function->handleRelesed(*wnd,it->second.overrideBehavior);
                    break;
                case GLFW_REPEAT:
                    it->second.function->handleRepeat(*wnd,it->second.overrideBehavior);
                    break;
                default:
                    break;
            }
        }
    }
}



}}}