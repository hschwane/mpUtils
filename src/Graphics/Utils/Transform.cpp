/*
 * mpUtils
 * transform.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the transform class
 *
 * Copyright (c) 2017 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Utils/Transform.h"
#include <glm/gtx/norm.hpp>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the transform class
//-------------------------------------------------------------------
Transform::Transform(const glm::vec3 position, const glm::quat orientation, const glm::vec3 scale)
        : position(position), scale(scale), orientation(orientation)
{

}

bool Transform::operator==(const Transform &other) const
{
    return position == other.position &&
           scale == other.scale &&
           orientation == other.orientation;
}

bool Transform::operator!=(const Transform &other) const
{
    return position != other.position ||
           scale != other.scale ||
           orientation != other.orientation;
}

Transform::Transform(const glm::mat4 mat)
{
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(mat, scale, orientation, position, skew, perspective);
    orientation = glm::conjugate(orientation);
}

Transform::operator glm::mat4() const
{
    return glm::translate(glm::mat4(1.f), position) * glm::scale(glm::mat4(1.f), scale) * glm::toMat4(orientation);
}

glm::quat rotationBetweenVectors(glm::vec3 start, glm::vec3 dest)
{
    // code from http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/
    start = normalize(start);
    dest = normalize(dest);
    glm::quat rot1;

    float cosTheta = dot(start, dest);
    glm::vec3 rotationAxis;

    if (cosTheta < -1 + 0.001f)
    {
        // special case when vectors in opposite directions:
        // there is no "ideal" rotation axis
        // So guess one; any will do as long as it's perpendicular to start
        rotationAxis = glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), start);
        if (glm::length2(rotationAxis) < 0.01 ) // bad luck, they were parallel, try again!
            rotationAxis = cross(glm::vec3(1.0f, 0.0f, 0.0f), start);

        rotationAxis = normalize(rotationAxis);
        return glm::angleAxis(glm::radians(180.0f), rotationAxis);
    }
    else
    {
        rotationAxis = cross(start, dest);

        float s = sqrt( (1+cosTheta)*2 );
        float invs = 1 / s;

        return glm::quat(
                s * 0.5f,
                rotationAxis.x * invs,
                rotationAxis.y * invs,
                rotationAxis.z * invs
        );
    }
}

glm::quat lookAt(glm::vec3 position, glm::vec3 target, glm::vec3 up)
{
    // code from http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/
    glm::vec3 direction = target - position;
    glm::quat rot1 = rotationBetweenVectors(glm::vec3(0.0f, 0.0f, 1.0f), direction);

    // Recompute desiredUp so that it's perpendicular to the direction
    // You can skip that part if you really want to force desiredUp
    glm::vec3 right = cross(direction, glm::normalize(up));
    up = cross(right, direction);

    // Because of the 1rst rotation, the up is probably completely screwed up.
    // Find the rotation between the "up" of the rotated object, and the desired up
    glm::vec3 newUp = rot1 * up;
    glm::quat rot2 = rotationBetweenVectors(newUp, up);

    return rot1 * rot2;
}

}}