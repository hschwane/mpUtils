/*
 * mpUtils
 * transform.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the transform class
 *
 * Copyright (c) 2017 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_TRANSFORM_H
#define MPUTILS_TRANSFORM_H

// includes
//--------------------
#include <glm/glm.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class transform
 *
 * usage:
 * A class which represents a full transformation including translation scale and orientation. It can be casted to glm::mat4
 * and can be explicit constructed out of one.
 *
 */
class Transform
{
public:
    // constructor
    explicit Transform(glm::mat4 mat);
    explicit Transform(glm::vec3 position={0,0,0}, glm::quat orientation = glm::angleAxis(0.f, glm::vec3(0)), glm::vec3 scale = {1, 1, 1});

    // comparision operators
    bool operator==(const Transform &other) const;
    bool operator!=(const Transform &other) const;

    // cast operators
    explicit operator glm::mat4() const;

    // public members for easy access
    glm::vec3 position{ 0, 0, 0 };
    glm::vec3 scale{ 1, 1, 1 };
    glm::quat orientation = glm::angleAxis(0.f, glm::vec3(0));
};

//-------------------------------------------------------------------
// some global functions to help with transform

/**
 * @brief calculates orientation between start and dest
 * @return quaternion representing the rotation
 */
glm::quat rotationBetweenVectors(glm::vec3 start, glm::vec3 dest);

/**
 * @brief orientation needed to look from position to target
 * @return the orientation one has to orient at position to look at target
 */
glm::quat lookAt(glm::vec3 position, glm::vec3 target, glm::vec3 up = {0.0f,1.0f,0.0f});



}}
#endif //MPUTILS_TRANSFORM_H
