/*
 * raptor
 * Camera.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Camera class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef RAPTOR_CAMERA_H
#define RAPTOR_CAMERA_H

// includes
//--------------------
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <cmath>
#include <string>
#include "mpUtils/Graphics/Utils/Transform.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Camera
 *
 * usage:
 *
 *
 */
class Camera
{
public:
    /**
     * @brief different camera modes, first person style and trackball style
     */
    enum CameraMode
    {
        trackball = 0,
        fps = 1
    };

    /**
     * @brief constructor of camera
     */
    Camera(CameraMode mode = fps, glm::vec3 position = glm::vec3(0,0,1), glm::vec3 target = glm::vec3(0, 0, 0), glm::vec3 world_up = glm::vec3(0, 1, 0), std::string uiName = "Camera");

    /**
     * Adds the following inputs to the input manager:
     *
     * RotateHorizontal : axis
     * RotateVertical : axis
     *
     * fps mode
     * MoveSideways : axis
     * MoveUpDown : axis
     * MoveForwardBackward : axis
     *
     * trackbal mode
     * Zoom : axis
     * PanHorizontal : axis
     * PanVertical : axis
     *
     * ToggleMode : button, toggles between fps and trackball mode
     * FastMode : button, activates fast mode for one frame when triggered
     * SlolwMode : button, activates fast mode for one frame when triggered
     * MovementSpeed : axis, changes movement and pan/zoom speed
     *
     * @param m_uiPrefix A prefix to be added to input manager ids. Default is "Camera".
     */
    void addInputs();
    void showDebugWindow(bool* show=nullptr); //!< shows an imgui window to debug camera things

    // update camera state
    void update(); //!< updates the cameras position and orientation based on rotate and move calls since the last update

    // functions to access camera coordinates / position, target and transformation
    glm::mat4 viewMatrix(); //!< get the view matrix to render something with this camers
    glm::mat4 modelMatrix(); //!< get the model matrix to render something at the cameras position

    glm::vec3 position() const {return m_transform.position;} //!< returns the camera position in cartesian coordinates
    glm::vec3 target() const {return m_transform.position + front() * m_targetDistance;} //!< returns point in space where camera is looking at / trackball is rotating around
    glm::quat orientation() const {return m_transform.orientation;} //!< returns the cameras orientation

    glm::vec3 up() const { return m_transform.orientation * glm::vec3(0,1,0); } //!< get up direction of this camera
    glm::vec3 down() const { return m_transform.orientation * glm::vec3(0,-1,0); } //!< get down direction of this camera
    glm::vec3 front() const { return m_transform.orientation * glm::vec3(0,0,-1); } //!< get forward direction of this camera
    glm::vec3 back() const { return m_transform.orientation * glm::vec3(0,0,1); } //!< get backward direction of this camera
    glm::vec3 right() const {return m_transform.orientation * glm::vec3(1,0,0); } //!< get right direction of this camera
    glm::vec3 left() const { return m_transform.orientation * glm::vec3(-1,0,0); } //!< get left direction of this camera

    // move and rotate the camera
    void rotateV(float dTheta); //!< rotate camera vertically
    void rotateH(float dPhi); //!< rotate camera horizontally
    void moveZ(float dz); //!< move in Z direction (forward/backward) in fps mode
    void moveX(float dx); //!< move camera in X direction (left/right) in fps mode
    void moveY(float dy); //!< pan camera in Y direction (up/down) in fps mode
    void panV(float dy); //!< pan camera vertical in trackball mode
    void panH(float dx); //!< pan camera horizontal in trackball mode
    void zoom(float dz); //!< zooom camera in trackball mode

    void setTarget(const glm::vec3& target);  //!< rotate the camera so it looks at target
    void setPosition(const glm::vec3& pos) {m_transform.position = pos;} //!< set a new position for the camera
    void setWorldUp(const glm::vec3& up){m_world_up = glm::normalize(up);} //!< change the direction the camera considers up

    // frame modifier functions
    void fastMode() { m_movementSpeedMod*=2.0f;} //!< doubles movement speed this frame
    void slowMode() {m_movementSpeedMod*=0.5f;} //!< halves movement speed this frame

    // settings
    void alwaysEnableAllControls(bool enable) {m_enableAllControls = enable;} //!< if true movement is enabled in trackball mode and pan/zoom in fps mode
    void setRotationSpeedFPS(float speed) { m_fpsRotationSpeed = speed;} //!< change rotation speed in fps mode
    void setRotationSpeedTB(float speed) { m_tbRotationSpeed = speed;} //!< change rotation speed in trackball mode
    void setMovementSpeed(float speed) {m_moveSpeed = speed;} //!< change movement speed in fps mode
    void setPanSpeed(float speed) {m_panSpeed = speed;} //!< change pan speed in trackball mode
    void setZoomSpeed(float speed) {m_zoomSpeed = speed;} //!< change zoom speed in trackball mode
    float getRotationSpeedFPS() { return m_fpsRotationSpeed;} //!< get rotation speed in fps mode
    float getRotationSpeedTB() { return m_tbRotationSpeed;} //!< get rotation speed in trackball mode
    float getMovementSpeed() { return m_moveSpeed;} //!< get movement speed in fps mode
    float getPanSpeed() { return m_panSpeed;} //!< get pan speed in trackball mode
    float getZoomSpeed() { return m_zoomSpeed;} //!< get zoom speed in trackball mode



    void setMode(CameraMode mode); //!< change the mode in which the camera is controlled
    CameraMode getMode(); //! check the current mode of the camera
    void toggleMode(); //!< toggles between different modes

private:

    // speed settings
    float m_fpsRotationSpeed{0.005}; //!< speed with which the camera rotates in fps mode
    float m_moveSpeed{0.125}; //!< speed with which the camera moves in fps mode
    float m_tbRotationSpeed{0.015}; //!< speed with which the camera rotates in trackball mode
    float m_panSpeed{0.006}; //!< speed with which the camera pans in trackball mode
    float m_zoomSpeed{0.3}; //!< speed with which the camera zooms (used for Z axis in trackball mode)
    float m_movementSpeedMod{1.0}; //!< temporary modified movement speed
    bool m_enableAllControls{false}; //!< enable movement while in trackball mode and pan/zoom while in fps mode

    // camera inputs
    glm::vec3 m_movementInput{0,0,0};
    glm::vec2 m_rotationInput{0,0};

    // camera state
    Transform m_transform; //!< transform object describing orientation and position of camera
    float m_targetDistance; //!< distance along the front vector where the center for trackball mode lives
    glm::vec3 m_world_up; //!< worlds up vector
    CameraMode m_mode; //!< current control mode of the camera

    // helper variables dependent on state
    glm::mat4 m_view; //!< view matrix is stored here after each update

    // ui and input
    std::string m_uiPrefix; //!< name of the camera shown in UI and Input mappings
};

}}

#endif //RAPTOR_CAMERA_H
