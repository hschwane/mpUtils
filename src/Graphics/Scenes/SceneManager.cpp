/*
 * mpUtils
 * SceneManager.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the SceneManager class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Scenes/SceneManager.h"
#include "mpUtils/Log/Log.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the SceneManager class
//-------------------------------------------------------------------
void SceneManager::update()
{
    if(m_currentScene)
        m_currentScene->update();
}

void SceneManager::draw()
{
    if(m_currentScene)
        m_currentScene->draw();
}

unsigned int SceneManager::addScene(std::unique_ptr<Scene> scene)
{
    m_lastUsedId++;
    m_scenes.insert(std::make_pair(m_lastUsedId,std::move(scene)));
    return m_lastUsedId;
}

void SceneManager::removeScene(unsigned int sceneId)
{
    auto it = m_scenes.find(sceneId);
    if(it != m_scenes.end())
    {
        if(m_currentScene == it->second.get())
        {
            m_currentScene = nullptr;
            m_currentSceneId = 0;
        }

        m_scenes.erase(it);
    } else
    {
        logWARNING("SceneManager") << "Attempt to remove scene that does not exist.";
    }
}

Scene* SceneManager::getScene(unsigned int id)
{
    return m_scenes[id].get();
}

void SceneManager::switchToScene(unsigned int id)
{
    if(m_currentScene)
        m_currentScene->onDeactivation();

    assert_critical(m_scenes.find(id) != m_scenes.end(),"SceneManager", "Attempt to switch to invalid scene id.");

    m_currentScene = m_scenes[id].get();
    m_currentSceneId = id;
    m_currentScene->onActivation();
}

}}