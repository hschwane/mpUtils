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

#include "mpUtils/Graphics/Window.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
namespace Input {
//--------------------

namespace {
    std::vector<Window*> m_wndList;
    Window* m_focusedWindow;
    Window* m_hoveredWindow;
}

void registerWindow(Window* wnd)
{
    m_wndList.push_back(wnd);
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
    glm::dvec2 p;
    glfwGetCursorPos(m_wndList[0]->window(),&p.x,&p.y);
    p += m_wndList[0]->getPosition();
    return p;
}

void setCursorScreenPos(double x, double y)
{
    auto wpos = m_wndList[0]->getPosition();
    glfwSetCursorPos(m_wndList[0]->window(), x - wpos.x, y-wpos.y);
}

void setCursorScreenPos(glm::dvec2 p)
{
    auto wpos = m_wndList[0]->getPosition();
    glfwSetCursorPos(m_wndList[0]->window(), p.x - wpos.x, p.y - wpos.y);
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
    glfwSetClipboardString(m_wndList[0]->window(),s.c_str());
}
std::string getClipboard()
{
    return std::string(glfwGetClipboardString(m_wndList[0]->window()));
}

}}}