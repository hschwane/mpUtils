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

void Transform::lookAt(const glm::vec3& target, const glm::vec3& up)
{
    orientation = glm::quatLookAt(glm::normalize(target-position),up);
}

}}