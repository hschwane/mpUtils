/*
 * raptor
 * Camera.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Camera class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include <glm/ext.hpp>
#include "mpUtils/Graphics/Rendering/Camera.h"
#include "mpUtils/Graphics/Input.h"
#include "mpUtils/Log/Log.h"
#include "mpUtils/Graphics/Gui/ImGui.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Camera class
//-------------------------------------------------------------------
Camera::Camera(CameraMode mode, glm::vec3 position, glm::vec3 target, glm::vec3 world_up, std::string uiName)
    : m_transform(position),
      m_desiredTransform(position),
      m_world_up(world_up),
      m_mode(mode),
      m_uiPrefix(std::move(uiName))
{
    setTarget(target);
    m_view = glm::inverse(static_cast<glm::mat4>(m_transform));
}

void Camera::addInputs()
{
    gph::Input::addAxis(m_uiPrefix + "RotateHorizontal", "Rotate camera horizontally.",
                        [this](gph::Window& wnd, double v)
                        {
                            this->rotateH(v);
                        });
    gph::Input::addAxis(m_uiPrefix + "RotateVertical", "Rotate camera vertically.",
                        [this](gph::Window& wnd, double v)
                        {
                            this->rotateV(v);
                        });

    gph::Input::addAxis(m_uiPrefix + "MoveSideways", "Move the camera sideways in FPS mode.",
                        [this](gph::Window& wnd, double v)
                        {
                            this->moveX(v);
                        });
    gph::Input::addAxis(m_uiPrefix + "MoveForwardBackward", "Move the camera forwards and backwards in FPS mode.",
                        [this](gph::Window& wnd, double v)
                        {
                            this->moveZ(v);
                        });
    gph::Input::addAxis(m_uiPrefix + "MoveUpDown", "Move the camera up and down in FPS mode.",
                        [this](gph::Window& wnd, double v)
                        {
                            this->moveY(v);
                        });

    gph::Input::addAxis(m_uiPrefix + "Zoom", "Zoom camera in and out in trackball mode.",
                        [this](gph::Window& wnd, double v)
                        {
                            this->zoom(v);
                        });
    gph::Input::addAxis(m_uiPrefix + "PanHorizontal", "Pan camera horizontally in trackball mode.",
                        [this](gph::Window& wnd, double v)
                        {
                            this->panH(v);
                        });
    gph::Input::addAxis(m_uiPrefix + "PanVertical", "Pan camera vertically in trackball mode.",
                        [this](gph::Window& wnd, double v)
                        {
                            this->panV(v);
                        });

    gph::Input::addButton(m_uiPrefix + "ToggleMode", "Toggles between fps and trackball mode.",
                          [this](gph::Window& wnd)
                          {
                              this->toggleMode();
                          });
    gph::Input::addButton(m_uiPrefix + "FastMode", "While triggered movement speed is doubled.",
                          [this](gph::Window& wnd)
                          {
                              this->fastMode();
                          });
    gph::Input::addButton(m_uiPrefix + "SlowMode", "While triggered movement speed is halved.",
                          [this](gph::Window& wnd)
                          {
                              this->slowMode();
                          });

    gph::Input::addAxis(m_uiPrefix + "MovementSpeed", "Change cameras movement and pan/zoom speed.",
                        [this](gph::Window& wnd, double v)
                        {
                                this->setMovementSpeed(  getMovementSpeed() + float(v) * 0.025f * (getMovementSpeed() + std::numeric_limits<float>::min()) );
                                this->setPanSpeed(  getPanSpeed() + float(v) * 0.025f * (getPanSpeed() + std::numeric_limits<float>::min()));
                                this->setZoomSpeed(  getZoomSpeed() + float(v) * 0.025f * (getZoomSpeed() + std::numeric_limits<float>::min()));
                        });
}

void Camera::setTarget(const glm::vec3& target, bool interpolate)
{
    m_desiredTransform.lookAt(target,m_world_up);
    m_desiredTargetDistance = glm::length(target-m_desiredTransform.position);

    if(!interpolate)
    {
        m_transform.orientation = m_desiredTransform.orientation;
        m_targetDistance = m_desiredTargetDistance;
    }
}

void Camera::setPosition(const glm::vec3& pos, bool interpolate)
{
    m_desiredTransform.position = pos;
    if(!interpolate)
        m_transform.position = pos;
}

void Camera::setMode(gph::Camera::CameraMode mode)
{
    m_mode = mode;
}

Camera::CameraMode Camera::getMode() const
{
    return m_mode;
}

void Camera::toggleMode()
{
    if(m_mode==CameraMode::trackball)
        setMode(CameraMode::fps);
    else
        setMode(CameraMode::trackball);
}

void Camera::update()
{
    // movement in camera coordinates
    glm::vec3 movement = m_transform.orientation * (m_movementInput * m_movementSpeedMod);

    if(m_mode == trackball)
    {
        // rotate position around target

        // figure out where the old target is
        glm::vec3 oldTarget = m_desiredTransform.position + m_desiredTransform.orientation * glm::vec3(0,0,-1) * m_desiredTargetDistance;

        // rotate the camera
        m_desiredTransform.orientation = glm::normalize( glm::angleAxis(m_rotationInput.x, m_world_up)
                                                  * m_desiredTransform.orientation
                                                  * glm::angleAxis(m_rotationInput.y, glm::vec3(1, 0, 0)) );

        // move so old target matches new target
        glm::vec3 newTarget = m_desiredTransform.position + m_desiredTransform.orientation * glm::vec3(0,0,-1) * m_desiredTargetDistance;
        m_desiredTransform.position += oldTarget - newTarget;

        // pan
        // zooming, make sure distance to the target does not become negative
        if(m_desiredTargetDistance + m_movementInput.z > 0.01f * m_zoomSpeed)
        {
            m_desiredTargetDistance += m_movementInput.z;
        }

        // now just apply movement
        m_desiredTransform.position += movement;

        // interpolate

        // interpolate distance to target
        m_targetDistance = glm::mix(m_targetDistance, m_desiredTargetDistance,
                                   static_cast<float>(glm::pow(Input::deltaTime(),m_movementSmoothing)));

        // interpolate current target position
        glm::vec3 desiredTarget = m_desiredTransform.position + m_desiredTransform.orientation * glm::vec3(0,0,-1) * m_desiredTargetDistance;
        glm::vec3 oldActualTarget = m_transform.position + m_transform.orientation * glm::vec3(0,0,-1) * m_targetDistance;
        oldActualTarget = glm::mix(oldActualTarget, desiredTarget,
                                   static_cast<float>(glm::pow(Input::deltaTime(),m_movementSmoothing)));

        // interpolate orientation
        m_transform.orientation = glm::slerp(m_transform.orientation, m_desiredTransform.orientation,
                                             static_cast<float>(glm::pow(Input::deltaTime(),m_rotationSmoothing)));

        // calculate proper position using difference that was created by rotation and moving the target
        glm::vec3 newActualTarget = m_transform.position + m_transform.orientation * glm::vec3(0,0,-1) * m_targetDistance;
        m_transform.position += oldActualTarget - newActualTarget;
    }
    else if(m_mode == fps)
    {
        // movement
        m_desiredTransform.position += movement;

        // rotation
        m_desiredTransform.orientation = glm::normalize( glm::angleAxis(m_rotationInput.x, m_world_up)
                                    * m_desiredTransform.orientation
                                    * glm::angleAxis(m_rotationInput.y, glm::vec3(1, 0, 0)) );

        // interpolate between transform and desiredtransform
        m_transform.position = glm::mix(m_transform.position, m_desiredTransform.position,
                                        static_cast<float>(glm::pow(Input::deltaTime(),m_movementSmoothing)));
        m_transform.orientation = glm::slerp(m_transform.orientation, m_desiredTransform.orientation,
                                             static_cast<float>(glm::pow(Input::deltaTime(),m_rotationSmoothing)));
    }

    m_view = glm::inverse(static_cast<glm::mat4>(m_transform));
    m_rotationInput = {0,0};
    m_movementInput = {0,0,0};
    m_movementSpeedMod = 1.0f;
}

void Camera::rotateH(float dPhi)
{
    m_rotationInput.x += dPhi * (m_mode==fps ? m_fpsRotationSpeed : m_tbRotationSpeed);
}

void Camera::rotateV(float dTheta)
{
    m_rotationInput.y += dTheta * (m_mode==fps ? m_fpsRotationSpeed : m_tbRotationSpeed);
}

void Camera::moveX(float dx)
{
    if(m_mode == fps || m_enableAllControls)
        m_movementInput.x += dx * m_moveSpeed;
}

void Camera::moveY(float dy)
{
    if(m_mode == fps || m_enableAllControls)
        m_movementInput.y += dy * m_moveSpeed;
}

void gph::Camera::moveZ(float dz)
{
    if(m_mode == fps || m_enableAllControls)
        m_movementInput.z -= dz * m_moveSpeed;
}

void Camera::panV(float dy)
{
    if(m_mode == trackball || m_enableAllControls)
        m_movementInput.y += dy * m_panSpeed;
}

void Camera::panH(float dx)
{
    if(m_mode == trackball || m_enableAllControls)
        m_movementInput.x += dx * m_panSpeed;
}

void Camera::zoom(float dz)
{
    if(m_mode == trackball || m_enableAllControls)
        m_movementInput.z -= dz * m_zoomSpeed;
}

glm::mat4 gph::Camera::viewMatrix() const
{
  return m_view;
}

glm::mat4 gph::Camera::modelMatrix() const
{
  return static_cast<glm::mat4>(m_transform);
}

void Camera::showDebugWindow(bool* show)
{
    if(ImGui::Begin((m_uiPrefix+std::string("Debug Information")).c_str(),show))
    {
        ImGui::InputFloat3("Desired Position",glm::value_ptr(m_desiredTransform.position));
        glm::vec3 desiredtarget = m_desiredTransform.position + front() * m_desiredTargetDistance;
        if( ImGui::InputFloat3(" Desire Target",glm::value_ptr(desiredtarget)))
        {
            m_desiredTransform.lookAt(desiredtarget,m_world_up);
            m_desiredTargetDistance = glm::length(desiredtarget-m_desiredTransform.position);
        }
        ImGui::InputFloat("Desired Distance to Target",&m_desiredTargetDistance);

        ImGui::Separator();

        ImGui::InputFloat3("Position",glm::value_ptr(m_transform.position));
        glm::vec3 target = m_transform.position + front() * m_targetDistance;
        if( ImGui::InputFloat3("Target",glm::value_ptr(target)))
        {
            m_transform.lookAt(target,m_world_up);
            m_targetDistance = glm::length(target-m_transform.position);
        }
        ImGui::InputFloat("Distance to Target",&m_targetDistance);
        ImGui::InputFloat3("World Up",glm::value_ptr(m_world_up));
        ImGui::Text("Orientation: %s",glm::to_string(m_transform.orientation).c_str());

        ImGui::Text("Mode: "); ImGui::SameLine();
        ImGui::RadioButton("trackball", reinterpret_cast<int*>(&m_mode), 0); ImGui::SameLine();
        ImGui::RadioButton("fps", reinterpret_cast<int*>(&m_mode), 1);

        static bool speed=false;
        static bool slow=false;
        bool speedEnabled = m_movementSpeedMod > 1.0f;
        bool slowEnabled = m_movementSpeedMod < 1.0f;
        if(speedEnabled)
        {
            ImGui::Checkbox("FastMode", &speedEnabled);
        }
        else
        {
            ImGui::Checkbox("FastMode", &speed);
            if(speed)
                fastMode();
        }
        ImGui::SameLine();
        if(slowEnabled)
        {
            ImGui::Checkbox("SlowMode", &slowEnabled);
        }
        else
        {
            ImGui::Checkbox("SlowMode", &slow);
            if(slow)
                slowMode();
        }

        ImGui::Checkbox("Enable all controls", &m_enableAllControls);

        ImGui::Separator();

        ImGui::Text("Sensitivity");
        ImGui::SliderFloat("RotateFPS",&m_fpsRotationSpeed,0.0005,0.1,"%.4f",2.0f);
        ImGui::SliderFloat("Move",&m_moveSpeed,0.005,1.0f,"%.4f",2.0f);
        ImGui::SliderFloat("RotateTB",&m_tbRotationSpeed,0.0005,0.1,"%.4f",2.0f);
        ImGui::SliderFloat("Pan",&m_panSpeed,0.001,.1,"%.4f",2.0f);
        ImGui::SliderFloat("Zoom",&m_zoomSpeed,0.01,2,"%.4f",2.0f);

        ImGui::Separator();

        ImGui::Text("Smoothing");
        ImGui::SliderFloat("Movement",&m_movementSmoothing,0,2,"%.3f",2.0f);
        ImGui::SliderFloat("Rotation",&m_rotationSmoothing,0,2,"%.3f",2.0f);

        ImGui::Separator();

        ImGui::Text("Camera coordinate System:");
        ImGui::Text("Front: %s \nBack:  %s \nRight: %s \nLeft:   %s \nUp:     %s\nDown: %s",
                    glm::to_string(front()).c_str(),
                    glm::to_string(back()).c_str(),
                    glm::to_string(right()).c_str(),
                    glm::to_string(left()).c_str(),
                    glm::to_string(up()).c_str(),
                    glm::to_string(down()).c_str());
    }
    ImGui::End();
}

}}