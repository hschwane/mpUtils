/*
 * mpUtils
 * Window.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Window class
 *
 * Copyright (c) 2017 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_WINDOW_H
#define MPUTILS_WINDOW_H

// includes
//--------------------
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Window
 *
 * usage:
 * This is an object oriented wrapper for glfw Window. Use the constructor to create the window, it will also make the context current.
 * Most glfw Window functions are implemented in the wrapper, however if you need a glfw Window you can use Window::window()
 * or a cast to obtain a pointer to a glfw Window.
 * To update the contents of your window use frameBegin and frameEnd. Inbetween you can draw things using openGL. BeginFrame will return false
 * if the operating system is asking the window to be closed. Clearing and swapping the framebuffer using (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) is also handled automatically.
 * You can change the mask with setClearMask().
 * The window is created with the window hints for openGL Version and core profile. To request a specific version call
 * Window::setGlVersion() before creating a window. If you want to use other special options, please use glfwSetWindowHint()
 * to configure all options to your liking before creating a window.
 * Callbacks support lambdas and function objects. Multiple callbacks of the same type can be added and will be called after each other.
 * When adding a callback a id is returned. Remove the callback using it's id before it becomes unsafe to call it! (Eg captured variables run out of scope)
 *
 * The Window class also implements the cutom callbacks frameBegin and frameEnd. FrameEnd is called just before buffers are swapped and
 * can be used to draw overlays, guis or measure performance. FrameBegin is called directly after glClear and can be used to perform
 * per frame initialization steps.
 *
 * While it is in general a good idea to use the input manager to handle inputs, some window specific settings and polling can be done
 * directly using thw window class.
 *
 */
class Window
{
public:

    // deal with gl version
    static void setGlVersion(int major, int minor); //!< change the opengl version you want (befor creating a window)
    static std::pair<int,int> getGlVersion(); //! returns the opengl version currently set

    // window creation helper
    static void setWindowHint(int hint, int value); //!< set glfw window hints (before creating a window)
    static void resetWindowHints(); //!< reset window hints to default
    static Window headlessContext(std::string title); //!< create an invisible window for headless rendering

    // constructor
    /**
     * @brief Create a new window. The created window needs still to be made the current context
     * @param width width of the window
     * @param height height of the window
     * @param title title of the window
     * @param monitor set a GLFWmonitor to create a fullscreen window
     * @param share supply a pointer to nother window in order to share one gl kontext
     */
    Window(int width, int height, const std::string & title, GLFWmonitor* monitor = nullptr, GLFWwindow* share = nullptr);

    /**
     * @brief creates window in fullscreen mode on monitor using current video settings
     * @param title title of the window
     * @param monitor monitor to create fullscreen window. if omitted primary monitor is used
     * @param share supply a pointer to another window in order to share one gl kontext
     */
    Window(const std::string & title, GLFWmonitor* monitor, GLFWwindow* share = nullptr);

    /**
     * @brief creates window in fullscreen mode on primary monitor using current video settings
     * @param title title of the window
     * @param share supply a pointer to another window in order to share one gl kontext
     */
    explicit Window(const std::string & title, GLFWwindow* share = nullptr);

    // destructor
    ~Window();

    // make noncopyable but moveable
    Window(const Window& that) = delete;
    Window& operator=(const Window& that) = delete;
    Window(Window&& that) = default;
    Window& operator=(Window&& that) = default;

    // access to internal glfw window
    explicit operator GLFWwindow*() const; //!< return the inner pointer to the glfw window
    GLFWwindow* window() const; //!< return the inner pointer to the glfw window

    // ------------
    // functions working with global state

    // context handling
    void makeContextCurrent() {glfwMakeContextCurrent(m_w.get());} //!< makes this window the current openGL context
    bool isContextCurrent() {return (this == getCurrentContext());} //!< is this windows context current?
    static Window* getCurrentContext(); //!< returns the window whose context is current or nullptr if no context is current
    static void pollEvents(){glfwPollEvents();} //!< manually poll for events when using the second update function with poll == false for all windows

    // ------------
    // functions working with window state

    /**
     * @brief Take care of things that need to be done at the start of a frame for this window. Also makes this windows
     *      openGL context current and calls all frameBegin callbacks. Returns false if the window should be closed.
     */
    bool frameBegin();
    /**
     * @brief Take care of things that need to be done at the end of a frame for this window. Also makes this window current
     * in case you handled another winddow inbetween frammeBegin and frameEnd and calls all frameEnd callbacks.
     */
    void frameEnd();

    // openGL functionality
    GLbitfield getClearMask() { return m_clearMask;} //!< get the mask that is passed to glClear default is (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    void setClearMask(GLbitfield clearMask) {m_clearMask = clearMask;} //!< set the mask that is passed to glClear default is (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    // functions handling fullscreen mode
    void makeFullscreen(int width, int height, GLFWmonitor* monitor); //!< make the window fullscreen
    void makeFullscreen(int width, int height); //!< make the window fullscreen on primary monitor
    void makeFullscreen(); //!< make the window fullscreen using current video settings on primary monitor
    void makeWindowed(); //!< change the window from fullscreen mode to windowed mode
    void toggleFullscreen(); //!< toggle between fullscreen and windowed mode
    bool isFullscreen() const {return (getMonitor() != nullptr);} //!< check if the window is a fullscreen window
    GLFWmonitor* getMonitor() const {return glfwGetWindowMonitor(m_w.get());} //!< returns the monitor the window uses for fullscreen mode

    // window setting functions
    void shouldClose() {glfwSetWindowShouldClose(m_w.get(),GLFW_TRUE);}  //!< signal the window to close
    void setTitle(const std::string & s) {glfwSetWindowTitle(m_w.get(),s.c_str());} //!< change the window title
    glm::ivec2 getPosition() const {glm::ivec2 p; glfwGetWindowPos(m_w.get(),&p.x,&p.y);return p;} //!< returns the window position
    void setPosition(glm::ivec2 pos) {glfwSetWindowPos(m_w.get(),pos.x,pos.y);} //!< sets a new window position
    void setPosition(int x, int y) {glfwSetWindowPos(m_w.get(),x,y);} //!< sets a new window position
    glm::ivec2 getSize() const {glm::ivec2 p; glfwGetWindowSize(m_w.get(),&p.x,&p.y);return p;} //!< returns the current window size
    glm::ivec2 getFramebufferSize() const {glm::ivec2 p; glfwGetFramebufferSize(m_w.get(),&p.x,&p.y);return p;} //!< returns the current window size
    void setSize(glm::ivec2 size) {glfwSetWindowSize(m_w.get(),size.x,size.y);} //!< resize the window
    void setSize(int x, int y) {glfwSetWindowSize(m_w.get(),x,y);} //!< resize the window
    void minimize() {glfwIconifyWindow(m_w.get());} //!< minimize the window
    void restore() {glfwRestoreWindow(m_w.get());} //!< restore a minimized window
    void toggleMinimize(); //!< toggle minimzation state
    bool isMinimized() const {return (GLFW_TRUE == glfwGetWindowAttrib(m_w.get(),GLFW_ICONIFIED));} //!< check if this the window is minimized
    void hide(){glfwHideWindow(m_w.get());} //!< make the window invisible
    void show(){glfwShowWindow(m_w.get());} //!< make the window visible if it was invisible before
    void toggleHide(); //!< toggle the visibility mode of the window
    bool isVisible() const {return (GLFW_TRUE == glfwGetWindowAttrib(m_w.get(),GLFW_VISIBLE));} //!< check if the window is visible
    void setIcon(int count, const GLFWimage* images); //!< set a list of images, the best one will be picked as the window icon
    GLFWmonitor* getWindowMonitor() const; //!< returns the monitor the bigger part of the window is currently on

    // input functions
    void setInputMode(int mode, int value) {glfwSetInputMode(m_w.get(),mode,value);} //!< see glfwSetInputMode for reference
    int getInputMode(int mode) const {return glfwGetInputMode(m_w.get(),mode);} //!< set glfwSetInputMode for reference
    bool isKeyDown(int key) const {return glfwGetKey(m_w.get(),key)==GLFW_PRESS;} //!< returns true if key is pressed
    bool isMouseButtonDown(int button) const {return glfwGetMouseButton(m_w.get(),button)==GLFW_PRESS;} //!< returns true if mouse button is pressed
    glm::dvec2 getCursorPos() const {glm::dvec2 p; glfwGetCursorPos(m_w.get(),&p.x,&p.y);return p;} //!< returns the cursor position within the window
    void setCursorPos(glm::dvec2 p) {glfwSetCursorPos(m_w.get(),p.x,p.y);} //!< sets a new cursor position
    void setCursorPos(double x, double y) {glfwSetCursorPos(m_w.get(),x,y);} //!< sets a new cursor position
    void setCursor(int shape) {setCursor(glfwCreateStandardCursor(shape));} //!< create and set a cursor with a standard shape
    void setCursor(GLFWcursor* c) {m_cursor=c; glfwSetCursor(m_w.get(),c);} //!< sets a new cursor
    void restoreCursor(){setCursor(m_cursor);} //!< restores a cursor after it was changed by direct call to the internal window (mainly needed for imgui)


    // ------------
    // callbacks

    // window handling callbacks
    int addPositionCallback(std::function<void(int,int)> f) {return addCallback(m_positionCallbacks,f);} //!< add a callback that is called whenever the window position is changed
    void removePositionCallback(int id) {removeCallback(m_positionCallbacks,id);} //!< removes the position callback function specified by id
    int addSizeCallback(std::function<void(int,int)> f) {return addCallback(m_sizeCallbacks,f);} //!< add a callback that is called whenever the window size is changed
    void removeSizeCallback(int id) {removeCallback(m_sizeCallbacks,id);}; //!< removes the size callback function specified by id
    int addCloseCallback(std::function<void()> f) {return addCallback(m_closeCallbacks,f);} //!< add a callback that is called whenever the window is closed
    void removeCloseCallback(int id) {removeCallback(m_closeCallbacks,id);}; //!< removes the close callback function specified by id
    int addRefreshRateCallback(std::function<void()> f) {return addCallback(m_refreshRateCallbacks,f);} //!< add a callback that is called whenever the window refresh rate is changed
    void removeRefreshRateCallback(int id) {removeCallback(m_refreshRateCallbacks,id);}; //!< removes the a refresh rate callback function specified by id
    int addFocusCallback(std::function<void(bool)> f) {return addCallback(m_focusCallbacks,f);} //!< add a callback that is called whenever the window looses or gains focus (true = gains focus)
    void removeFocusCallback(int id) {removeCallback(m_focusCallbacks,id);}; //!< removes the focus callback function specified by id
    int addMinimizeCallback(std::function<void(bool)> f) {return addCallback(m_minimizeCallbacks,f);} //!< add a callback that is called whenever the window  is minimized or restored (true = is minimized)
    void removeMinimizeCallback(int id) {removeCallback(m_minimizeCallbacks,id);}; //!< removes the minimize callback function specified by id
    int addFBSizeCallback(std::function<void(int,int)> f) {return addCallback(m_framebufferSizeCallbacks,f);} //!< add a callback that is called whenever the framebuffer size is changed
    void removeFBSizeCallback(int id) {removeCallback(m_framebufferSizeCallbacks,id);}; //!< removes the framebuffer resize function specified by id

    // frame begin / end callbacks
    int addFrameBeginCallback(std::function<void()> f) {return addCallback(m_frameBeginCallback,f);} //!< add a callback that is called at the beginning of every frame
    void removeFrameBeginCallback(int id) {removeCallback(m_frameBeginCallback,id);}; //!< removes the frameBegin callback function specified by id
    int addFrameEndCallback(std::function<void()> f) {return addCallback(m_frameEndCallback,f);} //!< add a callback that is called at the end of every frame
    void removeFrameEndCallback(int id) {removeCallback(m_frameEndCallback,id);}; //!< removes the frameEnd callback function specified by id

    // --------------
    // deprecated
    int getKey(int key) {return glfwGetKey(m_w.get(),key);} //!< ||deprecated|| state of key returns GLFW_PRESS or GLFW_RELEASE
    int getMouseButton(int button) {return glfwGetMouseButton(m_w.get(),button);} //!< ||deprecated|| state of mouse button returns GLFW_PRESS or GLFW_RELEASE
    // deprecated
    // --------------

private:
    static int gl_major; //!< major openGL version to use when creating the next window
    static int gl_minor; //!< minor openGL version to use when creating the next window

    template <typename T, typename F>
    int addCallback(std::vector<std::pair<int,T>>& callbackVector, F f); //!< helper to implement add callback functions

    template <typename T>
    void removeCallback(std::vector<std::pair<int,T>>& callbackVector, int id); //!< helper to implement remove callback functions

    std::unique_ptr<GLFWwindow,void(*)(GLFWwindow*)> m_w; //!< pointer to the glfw window

    // callback vectors for window functions
    std::vector<std::pair<int,std::function<void(int,int)>>> m_positionCallbacks;
    std::vector<std::pair<int,std::function<void(int,int)>>> m_sizeCallbacks;
    std::vector<std::pair<int,std::function<void()>>> m_closeCallbacks;
    std::vector<std::pair<int,std::function<void()>>> m_refreshRateCallbacks;
    std::vector<std::pair<int,std::function<void(bool)>>> m_focusCallbacks;
    std::vector<std::pair<int,std::function<void(bool)>>> m_minimizeCallbacks;
    std::vector<std::pair<int,std::function<void(int,int)>>> m_framebufferSizeCallbacks;

    // callback vectors for frame begin / end functions
    std::vector<std::pair<int,std::function<void()>>> m_frameBeginCallback;
    std::vector<std::pair<int,std::function<void()>>> m_frameEndCallback;

    // internal callbacks for window functions
    static void globalPositionCallback(GLFWwindow * window, int x, int y);
    static void globalSizeCallback(GLFWwindow * window, int w, int h);
    static void globalCloseCallback(GLFWwindow * window);
    static void globalRefreshRateCallback(GLFWwindow * window);
    static void globalFocusCallback(GLFWwindow * window, int f);
    static void globalMinimizeCalback(GLFWwindow * window, int m);
    static void globalFramebufferSizeCallback(GLFWwindow * window, int w, int h);

    glm::ivec2 m_origPos; //!< position before make fullscreen was called
    glm::ivec2 m_origSize; //!< size before make fullscreen was called
    GLbitfield m_clearMask; //!< passed to glClear
    GLFWcursor* m_cursor; //!< last manually set cursor
};

}}

//-------------------------------------------------------------------
// definitions of template functions of the window class

template<typename T, typename F>
int mpu::gph::Window::addCallback(std::vector<std::pair<int,T>> &callbackVector, F f)
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
void mpu::gph::Window::removeCallback(std::vector<std::pair<int,T>> &callbackVector, int id)
{
    auto it = std::lower_bound( callbackVector.cbegin(), callbackVector.cend(), std::pair<int,T>(id,T{}),
            [](const std::pair<int,T>& a, const std::pair<int,T>& b){return (a.first < b.first);});
    if(it != callbackVector.end())
        callbackVector.erase(it);
}


#endif //MPUTILS_WINDOW_H