/*
 * mpUtils
 * pointPicking.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Defines functions to help with point picking.
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 */


#ifndef MPUTILS_POINTPICKING_H
#define MPUTILS_POINTPICKING_H

// includes
//--------------------
#include <cmath>
#include <random>
//--------------------

// this file contains device/host functions that also need to compile when using gcc
//--------------------
#ifndef CUDAHOSTDEV
#ifdef __CUDACC__
        #define CUDAHOSTDEV __host__ __device__
    #else
        #define CUDAHOSTDEV
    #endif
#endif
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

/**
 * @brief returns a seed from a random number generator / pseudo random number generator (depending on your system)
 *          to be used to seed a pseudo random number generator like std::default_random_engine
 */
inline unsigned int getRanndomSeed()
{
    static std::random_device rd;
    return rd();
}

/**
 * @brief random position on the surface of a sphere with radius 1
 * @param u a uniform random value [0,1]
 * @param v a uniform random value [0,1]
 * @param catX resulting cartesian X-coordinate
 * @param catY resulting cartesian Y-coordinate
 * @param catZ resulting cartesian Z-coordinate
 */
CUDAHOSTDEV void randSphereShell(float u, float v, float& catX, float& catY, float& catZ);

/**
 * @brief random position inside a uniform density sphere
 * @param u a uniform random value [0,1]
 * @param v a uniform random value [0,1]
 * @param r a uniform random value [0,1]
 * @param radius the radius of the sphere
 * @param catX resulting cartesian X-coordinate
 * @param catY resulting cartesian Y-coordinate
 * @param catZ resulting cartesian Z-coordinate
 */
CUDAHOSTDEV void randUniformSphere(float u, float v, float r, float radius, float& catX, float& catY, float& catZ);

/**
 * @brief calculate values from the Halton Sequence
 * @param index control which number from the sequence is computed
 * @param base the base of the sequence, should be a prime
 * @return the "index"th number of the Halton Sequence with base "base"
 */
CUDAHOSTDEV float haltonSeq(int index, int base);


// implementation of point picking functions
//-------------------------------------------------------------------

CUDAHOSTDEV inline void randSphereShell(float u, float v, float &catX, float &catY, float &catZ)
{
    float theta = 2.0f * M_PI * u;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);

    float cosPhi = 2*v-1.f;
    float sinPhi = std::sqrt(std::fmax(0.0f, 1.0f- cosPhi*cosPhi));

    catX = cosTheta*sinPhi;
    catY = sinTheta*sinPhi;
    catZ = cosPhi;
}

CUDAHOSTDEV inline void randUniformSphere(float u, float v, float r, float radius, float &catX, float &catY, float &catZ)
{
    float theta = 2.f * M_PI * u;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);

    float cosPhi = 2*v-1.f;
    float sinPhi = std::sqrt(std::fmax(0.0f, 1.0f- cosPhi*cosPhi));

    r = std::pow(r,1.0f/3.0f) * radius;

    catX = r*cosTheta*sinPhi;
    catY = r*sinTheta*sinPhi;
    catZ = r*cosPhi;
}

CUDAHOSTDEV inline float haltonSeq(int index, int base)
{
    float f = 1;
    float r = 0;
    while(index > 0)
    {
        f = f/base;
        r = r + f* (index% base);
        index = index/base;
    }
    return r;
}

}

#endif //MPUTILS_POINTPICKING_H
