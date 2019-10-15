/*
 * mpUtils
 * SyncObject.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the SyncObject class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_SYNCOBJECT_H
#define MPUTILS_SYNCOBJECT_H

// includes
//--------------------
#include <GL/glew.h>
#include <utility>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class SyncObject
 *
 * usage:
 * When created a sync object is inserted in the openGL command queue it becomes ready when all openGL commands
 * issued before the sync object have been processed.
 * Use check ready to check if the sync object is ready or wait ready to wait until it becomes ready.
 * Don't forget to use glFlush, glSwapBuffers or set flush to true while waiting or the object might not get ready.
 *
 */
class SyncObject
{
public:
    SyncObject();
    ~SyncObject();

    // non copyable but movable
    SyncObject(const SyncObject& other) = delete;
    SyncObject& operator=(const SyncObject& other) = delete;
    SyncObject(SyncObject&& other) noexcept : m_syncObject(nullptr){*this = std::move(other);};
    SyncObject& operator=(SyncObject&& other) noexcept {using std::swap; swap(m_syncObject,other.m_syncObject); return *this;};

    /**
     * @brief check if the sync object is ready
     * @param flush should the command queue be flushed before checking?
     * @return returns true if the sync object is ready
     */
    bool checkReady(bool flush = false);

    /**
     * @brief wait timeout nanoseconds for the sync object to get ready
     * @param timeout amount of time to wait for the sync object to get ready (can wait longer)
     * @param flush should the command queue be flushed before checking?
     * @return true if the sync object is ready
     */
    bool waitReady(unsigned long timeout, bool flush = false);

private:
    GLsync m_syncObject;
};

}}

#endif //MPUTILS_SYNCOBJECT_H
