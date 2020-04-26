/*
 * mpUtils
 * Camera2D.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Camera2D class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include <glm/ext.hpp>
#include "mpUtils/Graphics/Rendering2D/Camera2D.h"
#include "mpUtils/Graphics/Input.h"
#include "mpUtils/Log/Log.h"
#include "mpUtils/Graphics/Gui/ImGui.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Camera2D class
//-------------------------------------------------------------------
Camera2D::Camera2D(glm::vec2 position, float zoom, std::string uiName)
    : m_currentTransform(position),
      m_currentZoom(zoom),
      m_desiredTransform(position),
      m_desiredZoom(zoom),
      m_uiPrefix(uiName)
{
}

void Camera2D::addInputs()
{

}

void Camera2D::showDebugWindow(bool* show)
{
    if(ImGui::Begin((m_uiPrefix+std::string(" Debug Information")).c_str(),show))
    {
        if(ImGui::CollapsingHeader("Movement"))
        {
            static glm::vec2 newPos(0, 0);
            static float newZoom = 1;
            static bool interpolate = false;
            ImGui::DragFloat2("##1", glm::value_ptr(newPos),0.1);
            ImGui::SameLine();
            if(ImGui::Button("set position"))
                setPosition(newPos, interpolate);
            ImGui::DragFloat("##2", &newZoom,0.1);
            ImGui::SameLine();
            if(ImGui::Button("set zoom"))
                setZoom(newZoom, interpolate);
            ImGui::Checkbox("Interpolate",&interpolate);

            ImGui::Separator();

            static glm::vec2 move(0,0);
            ImGui::DragFloat2("##4", glm::value_ptr(move),0.1);
            ImGui::SameLine();
            if(ImGui::Button("move"))
            {
                moveX(move.x);
                moveY(move.y);
            }

            static float zm=0;
            ImGui::DragFloat("##6", &zm,0.1);
            ImGui::SameLine();
            if(ImGui::Button("zoom"))
                zoom(zm);

            static float rot = 0;
            ImGui::DragFloat2("##3", &rot,0.1);
            ImGui::SameLine();
            if(ImGui::Button("rotate"))
                rotate(rot);
        }

        if(ImGui::CollapsingHeader("State"))
        {
            ImGui::Text("Position: %s",glm::to_string(m_currentTransform.position).c_str());
            ImGui::Text("Orientation: %f", m_currentTransform.orientation);
            ImGui::Text("Zoom: %f", m_currentZoom);

            static bool speed = false;
            static bool slow = false;
            bool speedEnabled = m_movementSpeedMod > 1.0f;
            bool slowEnabled = m_movementSpeedMod < 1.0f;
            if(speedEnabled)
            {
                ImGui::Checkbox("FastMode", &speedEnabled);
            } else
            {
                ImGui::Checkbox("FastMode", &speed);
                if(speed)
                    fastMode();
            }
            ImGui::SameLine();
            if(slowEnabled)
            {
                ImGui::Checkbox("SlowMode", &slowEnabled);
            } else
            {
                ImGui::Checkbox("SlowMode", &slow);
                if(slow)
                    slowMode();
            }
        }

        if(ImGui::CollapsingHeader("Sensitivity"))
        {
            ImGui::SliderFloat("Rotate", &m_rotationSpeed, 0.0005, 0.1, "%.4f", 2.0f);
            ImGui::SliderFloat("Move", &m_movementSpeed, 0.005, 1.0f, "%.4f", 2.0f);
            ImGui::SliderFloat("Zoom", &m_zoomSpeed, 0.01, 2, "%.4f", 2.0f);
        }

        if(ImGui::CollapsingHeader("Smoothing"))
        {
            ImGui::SliderFloat("Movement", &m_movementSmoothing, 0, 2, "%.3f", 2.0f);
            ImGui::SliderFloat("Rotation", &m_rotationSmoothing, 0, 2, "%.3f", 2.0f);
            ImGui::SliderFloat("Zoom", &m_zoomSmoothing, 0, 2, "%.3f", 2.0f);
        }
    }
    ImGui::End();
}

void Camera2D::update()
{
    // movement in camera orientation
    glm::vec2 movement = glm::rotate(m_movementInput,m_currentTransform.orientation) * m_movementSpeedMod;

    // add offsets to desired values
    m_desiredTransform.position += movement;
    m_desiredTransform.orientation += m_rotationInput;
    m_desiredZoom += m_zoomInput;

    // interpolate for smoothing
    m_currentTransform.position = glm::mix(m_currentTransform.position, m_desiredTransform.position,
                                    static_cast<float>(glm::pow(Input::deltaTime(),m_movementSmoothing)));
    m_currentZoom = glm::mix(m_currentZoom, m_desiredZoom,
            static_cast<float>(glm::pow(Input::deltaTime(),m_zoomSmoothing)));
    m_currentTransform.orientation = glm::mix(m_currentTransform.orientation, m_desiredTransform.orientation,
                             static_cast<float>(glm::pow(Input::deltaTime(),m_rotationSmoothing)));

    // create model and view mat
    m_model = static_cast<glm::mat4>(m_currentTransform);
    m_view = glm::inverse(glm::scale(glm::vec3(m_currentZoom)) * m_model);

    // reset data for next frame
    m_movementInput = {0,0};
    m_rotationInput = 0;
    m_zoomInput = 0;
    m_movementSpeedMod = 1.0f;
}

void Camera2D::rotate(float dPhi)
{
    m_rotationInput += dPhi * m_rotationSpeed;
}

void Camera2D::moveX(float dx)
{
    m_movementInput.x += dx * m_movementSpeed;
}

void Camera2D::moveY(float dy)
{
    m_movementInput.y += dy * m_movementSpeed;
}

void Camera2D::zoom(float dz)
{
    m_zoomInput += dz * m_zoomInput;
}

void Camera2D::setPosition(glm::vec2 pos, bool interpolate)
{
    m_desiredTransform.position = pos;
    if(!interpolate)
        m_currentTransform.position = pos;
}

void Camera2D::setZoom(float zoom, bool interpolate)
{
    m_desiredZoom = zoom;
    if(!interpolate)
        m_desiredZoom = zoom;
}


}}
