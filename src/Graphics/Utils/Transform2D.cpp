/*
 * mpUtils
 * Transform2D.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Transform2D class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Utils/Transform2D.h"
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Transform2D class
//-------------------------------------------------------------------
gph::Transform2D::Transform2D(glm::mat4 mat)
{
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::vec3 pos;
    glm::quat rot;
    glm::vec3 sc;
    glm::decompose(mat, sc, rot, pos, skew, perspective);

    position = glm::vec2(pos.x,pos.y);
    scale = glm::vec2(sc.x,sc.y);
    orientation = glm::eulerAngles(rot).z;
}

gph::Transform2D::Transform2D(glm::vec2 position, float orientation, glm::vec2 scale)
        : position(position), orientation(orientation), scale(scale)
{
}

bool gph::Transform2D::operator==(const Transform2D& other) const
{
    return position == other.position &&
           scale == other.scale &&
           orientation == other.orientation;
}

bool gph::Transform2D::operator!=(const Transform2D& other) const
{
    return position != other.position ||
           scale != other.scale ||
           orientation != other.orientation;
}

gph::Transform2D::operator glm::mat4() const
{
    glm::mat4 model(1);
    model =  glm::translate(model, glm::vec3(position,0.0f));
    model =  glm::rotate(model, orientation,glm::vec3{0.0f,0.0f,1.0f});
    model = glm::scale(model, glm::vec3(scale,0.0f));
    return model;
}

float angleBetweenVectors2D(glm::vec2 start, glm::vec2 dest)
{
    return glm::acos(glm::dot( glm::normalize(start), glm::normalize(dest)));
}

float lookAt2D(glm::vec2 position, glm::vec2 target)
{
    glm::vec2 direction = target - position;
    return angleBetweenVectors2D(glm::vec2(1, 0), direction) + (glm::sign(direction.x) < 0 ? glm::pi<float>() : 0.0f);
}

}}