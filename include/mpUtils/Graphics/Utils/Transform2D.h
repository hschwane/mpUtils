/*
 * mpUtils
 * Transform2D.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Transform2D class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_TRANSFORM2D_H
#define MPUTILS_TRANSFORM2D_H

// includes
//--------------------
#include <glm/glm.hpp>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Transform2D
 *
 * usage:
 * A class which represents a full transformation in 2D including translation scale and rotation. It can be casted to glm::mat4
 * and can be explicit constructed out of one.
 * Orientation 0 is in direction of x axis and increases counterclockwise.
 *
 */
class Transform2D
{
public:
    // constructor
    explicit Transform2D(glm::mat4 mat);
    explicit Transform2D(glm::vec2 position={0,0}, float rotation=0, glm::vec2 scale = {1, 1});

    // comparision operators
    bool operator==(const Transform2D& other) const;
    bool operator!=(const Transform2D& other) const;

    // cast operators
    explicit operator glm::mat4() const;

    // public members for easy access
    glm::vec2 position;
    glm::vec2 scale;
    float orientation;
};

//-------------------------------------------------------------------
// some global functions to help with transform

/**
 * @brief calculates angle between start and dest
 * @return angle between [0,pi]
 */
float angleBetweenVectors2D(glm::vec2 start, glm::vec2 dest);

/**
 * @brief orientation needed to look from position to target
 * @return the orientation one has to orient at position to look at target
 */
float lookAt2D(glm::vec2 position, glm::vec2 target);

}}

#endif //MPUTILS_TRANSFORM2D_H
