/*
 * mpUtils
 * Scene.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Scene class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_SCENE_H
#define MPUTILS_SCENE_H

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Scene
 *
 * usage: virtual base class for scenes, should be the parent of all scenes
 *
 */
class Scene
{
public:
    virtual ~Scene()=default;
    virtual void onActivation() =0; //!< will be called when scene is made active
    virtual void onDeactivation() =0; //!< will be called when scene is made deactive
    virtual void update() =0; //!< updates the scene
    virtual void draw() =0; //!< draws the scene
};

}}

#endif //MPUTILS_SCENE_H
