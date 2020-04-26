/*
 * mpUtils
 * Camera2D.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Camera2D class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_CAMERA2D_H
#define MPUTILS_CAMERA2D_H

// includes
//--------------------
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <cmath>
#include <string>
#include <mpUtils/Graphics/Utils/Transform2D.h>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Camera2D
 *
 * usage:
 *
 */
class Camera2D
{
public:
    /**
     * @brief create a 2d camera
     * @param position position to create the camera at
     * @param zoom zoom level of the camera
     */
    explicit Camera2D(glm::vec2 ={0,0}, float zoom = 1.0, std::string uiName = "Camera");

    /**
     * Adds the following inputs to the input manager:
     *
     */
    void addInputs();
    void showDebugWindow(bool* show=nullptr); //!< shows imgui window to debug camera

    // update camera
    void update(); //!< updates the camera state

    // access camera transformation
    glm::mat4 viewMatrix() const {return m_view;} //!< get the view matrix to render something with this camera
    glm::mat4 modelMatrix() const {return m_model;} //!< get the model matrix to render something at the cameras position

    // movement and rotation
    void rotate(float dPhi); //!< rotate camera around the down axis
    void moveX(float dx); //!< move camera in x direction
    void moveY(float dy); //!< move camera in y direction
    void zoom(float dz); //!< zoom the camera in and out

    void setPosition(glm::vec2 pos, bool interpolate = false); //!< set a new position for the camera if interpolate is true smoothing will be used
    void setZoom(float zoom, bool interpolate = false); //!< set a new zoom level for the camera if interpolate is true smoothing will be used

    // frame modifiers
    void fastMode() { m_movementSpeedMod*=2.0f;} //!< doubles movement speed this frame
    void slowMode() {m_movementSpeedMod*=0.5f;} //!< halves movement speed this frame

    // settings
    void setMovementSpeed(float speed) {m_movementSpeed = speed;} //!< change movement speed
    void setRotationSpeed(float speed) {m_rotationSpeed = speed;} //!< change rotation speed
    void setZoomSpeed(float speed) {m_zoomSpeed = speed;} //!< change zoom speed
    void setMovementSmoothing(float smooth) {m_movementSmoothing = smooth;} //!< set the smoothing strength for movement
    void setRotationSmoothing(float smooth) {m_movementSmoothing = smooth;} //!< set the smoothing strength for rotation
    void setZoomSmoothing(float smooth) {m_zoomSmoothing = smooth;} //!< set the smoothing strength for zoom
    float getMovementSpeed() const { return m_movementSpeed;} //!< get movement speed
    float getRotationSpeed() const { return m_rotationSpeed;} //!< get rotation speed
    float getZoomSpeed() const { return m_zoomSpeed;} //!< get zoom speed
    float getMovementSmoothing() const {return m_movementSmoothing;} //!< get the smoothing strength for movement
    float getRotationSmoothing() const {return m_rotationSmoothing;} //!< get the smoothing strength for rotation
    float getZoomSmoothing() const {return m_zoomSmoothing;} //!< get the smoothing strength for zoom

private:
    // settings
    float m_movementSpeedMod{1.0f}; //!< temporary modified movement speed
    float m_movementSpeed{0.125}; //!< speed with which the camera moves
    float m_rotationSpeed{0.1}; //!< speed with which the camera rotates
    float m_zoomSpeed{0.1}; //!< speed with which the camera zooms
    float m_movementSmoothing{0.25}; //!< higher numbers result in stronger smoothing and delayed movement
    float m_zoomSmoothing{0.25}; //!< higher numbers result in stronger smoothing and delayed zoom
    float m_rotationSmoothing{0.3}; //!< higher numbers result in stronger smoothing and delayed rotation

    // state
    Transform2D m_currentTransform; //!< current transformation
    Transform2D m_desiredTransform; //!< desired transformation based on inputs
    float m_currentZoom; //!< current zoom level
    float m_desiredZoom; //!< desired zoom level based on inputs

    // inputs
    glm::vec2 m_movementInput{0,0};
    float m_rotationInput{0};
    float m_zoomInput{0};

    // helper variables dependent on state
    glm::mat4 m_view{1.0};
    glm::mat4 m_model{1.0};

    // ui and input
    std::string m_uiPrefix; //!< name of the camera shown in UI and Input mappings
};

}}
#endif //MPUTILS_CAMERA2D_H
