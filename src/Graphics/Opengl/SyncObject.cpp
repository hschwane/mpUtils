/*
 * mpUtils
 * SyncObject.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the SyncObject class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Opengl/SyncObject.h"
#include "mpUtils/Log/Log.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the SyncObject class
//-------------------------------------------------------------------
SyncObject::SyncObject()
{
    m_syncObject = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
}

SyncObject::~SyncObject()
{
    glDeleteSync(m_syncObject);
}

bool SyncObject::checkReady(bool flush)
{
    return waitReady(0,flush);
}

bool SyncObject::waitReady(unsigned long timeout, bool flush)
{
    GLenum result = glClientWaitSync(m_syncObject, flush ? GL_SYNC_FLUSH_COMMANDS_BIT : 0, timeout);

    if(result==GL_ALREADY_SIGNALED || result==GL_CONDITION_SATISFIED)
        return true;
    else if(result==GL_TIMEOUT_EXPIRED)
        return false;
    else
    {
        logERROR("SyncObject") << "Error while waiting on sync object.";
        logFlush();
        throw std::runtime_error("Error while waiting on sync object");
    }
}

}}