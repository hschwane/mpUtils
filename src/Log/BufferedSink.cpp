/*
 * mpUtils
 * BufferedSink.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the BufferedSink class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include <future>
#include "mpUtils/Log/BufferedSink.h"
#include "mpUtils/Misc/timeUtils.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------


// function definitions of the BufferedSink class
//-------------------------------------------------------------------
LogBuffer::LogBuffer(int initialCapacity)
    : m_data(initialCapacity+1), m_readLine(0), m_insertLine(0)
{
}

void LogBuffer::addLine(LogMessage line)
{
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_changeBufferMtx);

    int localInsertLine = m_insertLine.load();
    int localReadLine = m_readLine.load();
    int p = localInsertLine;

    localInsertLine=(1+localInsertLine)%m_data.size();
    if(localInsertLine == localReadLine)
        localReadLine = (localReadLine+1)%m_data.size();

    m_readLine.store(localReadLine);
    m_insertLine.store(localInsertLine);

    m_data[p] = line;

    if(!m_rebuildingTheFilterAsync && checkFilter(m_data[p]))
    {
        m_filtered.emplace_back(m_data[p]);
        m_newFilterState = true;

        if(filteredSize() > size())
            rebuildFilter();
    }
    m_newMessage = true;
}

LogMessage& LogBuffer::operator[](int i)
{
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_changeBufferMtx);
    size_t idx = (m_readLine+i)%m_data.size();
    return m_data[idx];
}

void LogBuffer::changeCapacity(int newCap)
{
    std::unique_lock<std::shared_timed_mutex> lck(m_changeBufferMtx);
    m_data.resize(newCap+1);
    if(m_readLine >= newCap+1)
        m_readLine = newCap-2;
    if(m_insertLine >= newCap+1)
        m_insertLine = newCap-1;
    if(m_readLine > m_insertLine)
        m_readLine = 0;

    lck.unlock();
    rebuildFilter();
}

void LogBuffer::clear()
{
    m_pauseRebuilding = true;
    m_restartRebuilding = true;
    std::unique_lock<std::shared_timed_mutex> lck(m_changeBufferMtx);
    size_t size = m_data.size();
    m_data.clear();
    m_data.resize(size);
    m_readLine =0;
    m_insertLine=0;
    m_filtered.clear();
    m_pauseRebuilding = false;
}

bool LogBuffer::hasNewMessages()
{
    if(m_newMessage)
    {
        m_newMessage = false;
        return true;
    }
    return false;
}

int LogBuffer::capacity()
{
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_changeBufferMtx);
    return m_data.size()-1;
}

bool LogBuffer::full()
{
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_changeBufferMtx);
    return (m_insertLine+1)%m_data.size() == m_readLine;
}

bool LogBuffer::empty()
{
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_changeBufferMtx);
    return m_insertLine == m_readLine;
}

int LogBuffer::size()
{
    std::shared_lock<std::shared_timed_mutex> sharedLck(m_changeBufferMtx);
    if(m_insertLine >= m_readLine)
        return m_insertLine - m_readLine;
    else
        return m_data.size() + m_insertLine - m_readLine;
}

void LogBuffer::setMessageFilter(std::string filter)
{
    m_messageFilter = std::move(filter);
    rebuildFilter();
}

void LogBuffer::setAllowedLogLevels(std::array<bool, 7> lvls)
{
    m_allowedLogLvls = lvls;
    rebuildFilter();
}

void LogBuffer::setModuleFilter(std::string filter)
{
    m_moduleFilter = std::move(filter);
    rebuildFilter();
}

void LogBuffer::setFileFilter(std::string filter)
{
    m_fileFilter = std::move(filter);
    rebuildFilter();
}

void LogBuffer::setThreadFilter(std::thread::id id)
{
    m_tidFilter = id;
    rebuildFilter();
}

bool LogBuffer::filterChanged()
{
    if(m_newFilterState)
    {
        m_newFilterState = false;
        return true;
    }
    return false;
}

LogMessage& LogBuffer::filtered(int i)
{
    return m_filtered[i];
}

int LogBuffer::filteredSize()
{
    return m_filtered.size();
}

void LogBuffer::rebuildFilter()
{
    if(size() < 500 && !m_rebuildingTheFilterAsync)
    {
        m_filtered.clear();
        for(int i = 0; i < size(); i++)
        {
            if(checkFilter((*this)[i]))
            {
                m_filtered.emplace_back((*this)[i]);
            }
        }
        m_newFilterState = true;
    }
    else if(!m_rebuildingTheFilterAsync)
    {
        m_filtered.clear();
        m_rebuildingTheFilterAsync = true;
        static std::future<void> r;
        // for this many lines better filter async to prevent framerate drop
        r = std::async([&]()
        {
            for(int i = 0; i < size(); i++)
            {
                while(m_pauseRebuilding)
                    mpu::yield();

                if(m_restartRebuilding)
                {
                    m_restartRebuilding = false;
                    i=0;
                    continue;
                }

                if(checkFilter((*this)[i]))
                {
                    m_filtered.emplace_back( (*this)[i] );
                    m_newFilterState = true;
                }
            }
            m_rebuildingTheFilterAsync = false;
        });
    } else
    {
        m_pauseRebuilding = true;
        m_restartRebuilding = true;
        m_filtered.clear();
        m_pauseRebuilding = false;
    }
}

bool LogBuffer::checkFilter(const LogMessage& msg)
{
    bool passesFilter = m_allowedLogLvls[ (msg.lvl < 0 || msg.lvl >7) ? 0 : msg.lvl ];

    if(passesFilter && !m_moduleFilter.empty())
    {
        if(m_moduleFilter[0] == '-')
            passesFilter = (msg.sModule.find(m_moduleFilter.c_str()+1) == std::string::npos);
        else
            passesFilter = (msg.sModule.find(m_moduleFilter,0) != std::string::npos);
    }

    if(passesFilter && !m_messageFilter.empty())
    {
        if(m_messageFilter[0] == '-')
            passesFilter = (msg.sMessage.find(m_messageFilter.c_str()+1) == std::string::npos);
        else
            passesFilter = (msg.sMessage.find(m_messageFilter,0) != std::string::npos);
    }

    if(passesFilter && !(m_tidFilter == std::thread::id()))
        passesFilter = (m_tidFilter == msg.threadId);

    if(passesFilter && !m_fileFilter.empty())
    {
        if(m_fileFilter[0] == '-')
            passesFilter = (msg.sFilePosition.find(m_fileFilter.c_str()+1) == std::string::npos);
        else
            passesFilter = (msg.sFilePosition.find(m_fileFilter,0) != std::string::npos);
    }

    return passesFilter;
}

BufferedSink::BufferedSink(LogBuffer& buffer)
        : m_buffer(buffer)
{
}

void BufferedSink::operator()(const LogMessage& msg)
{
    const std::string& str = msg.sMessage;

    bool forcePlaintext = false;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find_first_of("\n\r", prev)) != std::string::npos)
    {
        m_buffer.addLine({str.substr(prev, pos - prev),msg.sFilePosition, msg.sModule, msg.lvl, msg.timepoint, msg.threadId, msg.plaintext});
        forcePlaintext = true;
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    m_buffer.addLine({str.substr(prev, pos - prev),msg.sFilePosition, msg.sModule, msg.lvl, msg.timepoint, msg.threadId, forcePlaintext || msg.plaintext});
}

}