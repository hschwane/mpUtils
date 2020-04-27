/*
 * mpUtils
 * SceneManager.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the SceneManager class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_SCENEMANAGER_H
#define MPUTILS_SCENEMANAGER_H

// includes
//--------------------
#include <memory>
#include <unordered_map>
#include "Scene.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class SceneManager
 *
 * Stores Scenes and executes the currently active scene.
 *
 */
class SceneManager
{
public:
    SceneManager()=default;

    void update(); //!< updates the current scene
    void draw(); //!< draws the current scene

    unsigned int addScene(std::unique_ptr<Scene> scene); //!< adds a scene to the manager
    void removeScene(unsigned int sceneId); //!< removes a scene from the manager

    Scene* getScene(unsigned int id); //!< returns pointer to a given scene (be careful!)
    void switchToScene(unsigned int id); //!< change to scene with id

    unsigned int getCurrentSceneId() {return m_currentSceneId;} //!< get id of currently active scene

private:
    Scene* m_currentScene{nullptr};
    unsigned int m_currentSceneId{0};
    std::unordered_map<unsigned int, std::unique_ptr<Scene>> m_scenes;
    unsigned int m_lastUsedId{0};
};

}}
#endif //MPUTILS_SCENEMANAGER_H
