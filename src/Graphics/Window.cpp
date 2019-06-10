/*
 * mpUtils
 * Window.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Window class
 *
 * Copyright (c) 2017 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Log/Log.h"
#include "mpUtils/Graphics/Window.h"
#include <cmath>
#include "Graphics/InputInternal.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definition for the glDebug callback
//-------------------------------------------------------------------
/**
 * @brief callback function used internally to write openGL errors to the log
 */
static void glDebugCallback(GLenum source, GLenum type, GLuint id, const GLenum severity, GLsizei length, const GLchar* message, const void* user_param)
{
    const auto format_message = [&] {
        return "source=\"" + [&]() -> std::string {
            switch (source)
            {
                case GL_DEBUG_SOURCE_API: return "API";
                case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "Window System";
                case GL_DEBUG_SOURCE_SHADER_COMPILER: return "Shader Compiler";
                case GL_DEBUG_SOURCE_THIRD_PARTY: return "Third Party";
                case GL_DEBUG_SOURCE_APPLICATION: return "Application";
                default: return "Other";
            }
        }() + "\", type=\"" + [&]() -> std::string {
            switch (type)
            {
                case GL_DEBUG_TYPE_ERROR: return "Error";
                case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "Deprecated Behavior";
                case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "Undefined Behavior";
                case GL_DEBUG_TYPE_PORTABILITY: return "Portability";
                case GL_DEBUG_TYPE_PERFORMANCE: return "Performance";
                case GL_DEBUG_TYPE_MARKER: return "Marker";
                case GL_DEBUG_TYPE_PUSH_GROUP: return "Push Group";
                case GL_DEBUG_TYPE_POP_GROUP: return "Pop Group";
                default: return "Other";
            }
        }() + "\" -- " + message;
    };

    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:
            logERROR("OpenGL") << format_message();
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            logWARNING("OpenGL") << format_message();
            break;
        case GL_DEBUG_SEVERITY_LOW:
            logINFO("OpenGL") << format_message();
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            logDEBUG("OpenGL") << format_message();
            break;
        default:
            break;
    }
}

// function definitions of the Window class
//-------------------------------------------------------------------
Window::Window(const int width, const int height, const std::string &title, GLFWmonitor *monitor, GLFWwindow *share)
    : m_w(nullptr,[](GLFWwindow* wnd){}), m_origPos(0,0), m_origSize(width-5,height-100),
    m_clearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT), m_cursor(nullptr)
{
    // init glfw once
    static struct GLFWinit
    {
        GLFWinit()
        {
            int e =glfwInit();
            if(e  != GL_TRUE)
            {
                logFATAL_ERROR("Graphics") << "Error initializing glfw. Returned: " << e ;
                throw std::runtime_error("Could not initializing glfw!");
            }

            glfwSetErrorCallback([](int code, const char * message){
                logERROR("GLFW") << "Error code: " << code << "Message: " << message;
            });
            logDEBUG("Graphics") << "initialized GLFW.";
        }
        ~GLFWinit() { glfwTerminate(); }
    } glfwinit;

    // setting some important default settings
#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl_major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl_minor);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* wnd = glfwCreateWindow(width, height, title.data(), monitor, share);
    if(!wnd)
    {
        logERROR("Window") << "Cannot create window.";
        throw std::runtime_error("Cannot create window");
    }
    m_w = std::unique_ptr<GLFWwindow,void(*)(GLFWwindow*)>(wnd,[](GLFWwindow* wnd){glfwDestroyWindow(wnd);});
    glfwSetWindowUserPointer(m_w.get(),this);
    makeContextCurrent();

    // attach the window handling callbacks
    glfwSetWindowPosCallback(m_w.get(),globalPositionCallback);
    glfwSetWindowSizeCallback(m_w.get(),globalSizeCallback);
    glfwSetWindowCloseCallback(m_w.get(),globalCloseCallback);
    glfwSetWindowRefreshCallback(m_w.get(),globalRefreshRateCallback);
    glfwSetWindowFocusCallback(m_w.get(),globalFocusCallback);
    glfwSetWindowIconifyCallback(m_w.get(),globalMinimizeCalback);
    glfwSetFramebufferSizeCallback(m_w.get(),globalFramebufferSizeCallback);

    // register to the input manager
    Input::registerWindow(this);

    // init glew
    static struct GLEWinit
    {
        GLEWinit() {
            glewExperimental = GL_TRUE;
            GLenum e = glewInit();
            if(e != GLEW_OK)
            {
                logFATAL_ERROR("Graphics") << "Error initializing glew. Returned: " << e ;
                throw std::runtime_error("Could not initialized glew!");
            }
            logDEBUG("Graphics") << "Initialised GLEW.";
        }
    } glewinit;

    glDebugMessageCallback(&glDebugCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, false);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW, 0, nullptr, false);
}

Window::Window(const std::string &title, GLFWmonitor *monitor, GLFWwindow *share)
    : Window(glfwGetVideoMode(monitor)->width, glfwGetVideoMode(monitor)->height,
            title, monitor, share)
{
}

Window::Window(const std::string &title, GLFWwindow *share)
    : Window(title,glfwGetPrimaryMonitor(),share)
{
}

Window::~Window()
{
    Input::unregisterWindow(this);
}

Window::operator GLFWwindow*() const
{
    return m_w.get();
}

GLFWwindow* Window::window() const
{
    return m_w.get();
}

int Window::gl_major = 4;
int Window::gl_minor = 5;
void Window::setGlVersion(int major, int minor)
{
    gl_major = major;
    gl_minor = minor;
}

void Window::setWindowHint(int hint, int value)
{
    glfwWindowHint(hint, value);
}

Window Window::headlessContext(std::string title)
{
    mpu::gph::Window::setWindowHint(GLFW_VISIBLE,false);
    return std::move(mpu::gph::Window(5,5,title));
}

bool Window::update()
{
    // end previous frame
    if(!isContextCurrent())
        makeContextCurrent();

    for(const auto &callback : m_frameEndCallback)
    {
        callback.second();
    }
    glfwSwapBuffers(m_w.get());

    //---------------
    // start next frame
    glfwPollEvents();

    // check if window needs to close
    if(glfwWindowShouldClose(m_w.get()))
        return false;

    glClear(m_clearMask);

    // call frame begin callbacks
    for(const auto &callback : m_frameBeginCallback)
    {
        callback.second();
    }

    return true;
}

bool Window::update(bool poll)
{
    // end previous frame
    if(!isContextCurrent())
        makeContextCurrent();

    for(const auto &callback : m_frameEndCallback)
    {
        callback.second();
    }
    glfwSwapBuffers(m_w.get());

    //---------------
    // start next frame
    if(poll)
        glfwPollEvents();

    // check if window needs to close
    if(glfwWindowShouldClose(m_w.get()))
        return false;

    glClear(m_clearMask);

    // call frame begin callbacks
    for(const auto &callback : m_frameBeginCallback)
    {
        callback.second();
    }

    return true;
}

std::pair<int, int> Window::getGlVersion()
{
    return std::pair<int, int>(gl_major,gl_minor);
}

Window *Window::getCurrentContext()
{
    auto wnd = glfwGetCurrentContext();
    if(wnd)
        return static_cast<Window*>(glfwGetWindowUserPointer(wnd));
    else
        return nullptr;
}

void Window::makeFullscreen(const int width, const int height, GLFWmonitor *monitor)
{
    m_origPos = getPosition();
    m_origSize = getSize();
    glfwSetWindowMonitor(m_w.get(), monitor, 0,0, width, height, GLFW_DONT_CARE);
}

void Window::makeFullscreen(const int width, const int height)
{
    makeFullscreen(width,height, getWindowMonitor());
}

void Window::makeFullscreen()
{
    auto vm = glfwGetVideoMode(getWindowMonitor());
    makeFullscreen(vm->width,vm->height);
}

void Window::makeWindowed()
{
    glfwSetWindowMonitor(m_w.get(), nullptr, m_origPos.x, m_origPos.y, m_origSize.x, m_origSize.y, GLFW_DONT_CARE);
}

void Window::toggleFullscreen()
{
    if(isFullscreen())
        makeWindowed();
    else
        makeFullscreen();
}

void Window::toogleMinimize()
{
    if(isMinimized())
        restore();
    else
        minimize();
}

void Window::toggleHide()
{
    if(isVisible())
        hide();
    else
        show();
}

void Window::setIcon(int count, const GLFWimage *images)
{
    glfwSetWindowIcon(m_w.get(), count, images);
}

GLFWmonitor *Window::getWindowMonitor()
{
    int nmonitors, i;
    int wx, wy, ww, wh;
    int mx, my, mw, mh;
    int overlap, bestoverlap;
    GLFWmonitor *bestmonitor;
    GLFWmonitor **monitors;
    const GLFWvidmode *mode;

    bestoverlap = 0;
    bestmonitor = NULL;

    glfwGetWindowPos(window(), &wx, &wy);
    glfwGetWindowSize(window(), &ww, &wh);
    monitors = glfwGetMonitors(&nmonitors);

    for (i = 0; i < nmonitors; i++) {
        mode = glfwGetVideoMode(monitors[i]);
        glfwGetMonitorPos(monitors[i], &mx, &my);
        mw = mode->width;
        mh = mode->height;

        overlap =
                std::max(0, std::min(wx + ww, mx + mw) - std::max(wx, mx)) *
                std::max(0, std::min(wy + wh, my + mh) - std::max(wy, my));

        if (bestoverlap < overlap) {
            bestoverlap = overlap;
            bestmonitor = monitors[i];
        }
    }

    return bestmonitor;
}

void Window::globalPositionCallback(GLFWwindow * window, int x, int y)
{
    Window* windowObject = static_cast<Window*>(glfwGetWindowUserPointer(window));
    for(const auto &callback : windowObject->m_positionCallbacks)
    {
        callback.second(x,y);
    }
}

void Window::globalSizeCallback(GLFWwindow *window, int w, int h)
{
    Window* windowObject = static_cast<Window*>(glfwGetWindowUserPointer(window));
    for(const auto &callback : windowObject->m_sizeCallbacks)
    {
        callback.second(w,h);
    }
}

void Window::globalCloseCallback(GLFWwindow *window)
{
    Window* windowObject = static_cast<Window*>(glfwGetWindowUserPointer(window));
    for(const auto &callback : windowObject->m_closeCallbacks)
    {
        callback.second();
    }
}

void Window::globalRefreshRateCallback(GLFWwindow *window)
{
    Window* windowObject = static_cast<Window*>(glfwGetWindowUserPointer(window));
    for(const auto &callback : windowObject->m_refreshRateCallbacks)
    {
        callback.second();
    }
}

void Window::globalFocusCallback(GLFWwindow *window, int f)
{
    Window* windowObject = static_cast<Window*>(glfwGetWindowUserPointer(window));
    for(const auto &callback : windowObject->m_focusCallbacks)
    {
        callback.second((f==GLFW_TRUE));
    }
}

void Window::globalMinimizeCalback(GLFWwindow *window, int m)
{
    Window* windowObject = static_cast<Window*>(glfwGetWindowUserPointer(window));
    for(const auto &callback : windowObject->m_minimizeCallbacks)
    {
        callback.second((m==GLFW_TRUE));
    }
}

void Window::globalFramebufferSizeCallback(GLFWwindow *window, int w, int h)
{
    Window* windowObject = static_cast<Window*>(glfwGetWindowUserPointer(window));
    for(const auto &callback : windowObject->m_framebufferSizeCallbacks)
    {
        callback.second(w,h);
    }
}

}}
