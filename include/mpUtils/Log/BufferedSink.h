/*
 * mpUtils
 * BufferedSink.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the BufferedSink class
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_BUFFEREDSINK_H
#define MPUTILS_BUFFEREDSINK_H

// includes
//--------------------
#include "Log.h"
#include <string>
#include <shared_mutex>
#include <atomic>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
/**
 * class LogBuffer
 *
 * The BufferedSink can add messages to the Buffer, so they can later be processed. Elements can also be accesed in filtered mode.
 * Filter is updated asyncronously so it can be changes often with out causing drop in fps.
 *
 */
class LogBuffer
{
public:
    explicit LogBuffer(int initialCapacity);
    void addLine(LogMessage line); //!< add a line to the buffer

    // filter
    void setAllowedLogLevels(std::array<bool,7> lvls); //!< the alowed log levels, [0] is other levels, [1] fatal, [2] error, etc
    void setMessageFilter(std::string filter); //!< filter by messages according to filter (include,-exclude)
    void setModuleFilter(std::string filter); //!< filter by module according to filter (include,-exclude)
    void setThreadFilter(std::thread::id id); //!< only show messages from specific thread (use std::thread::id() to show all threads)
    void setFileFilter(std::string filter); //!< filter by filr according to filter (include,-exclude)

    // clear
    void clear(); //!< clear the buffer

    // filtered element acces
    LogMessage& filtered(int i); //!< access element of filtered data
    int filteredSize(); //!< number of lines shown with filter
    bool filterChanged(); //!< checks if the list of filtered messages was changed

    // element access
    LogMessage& operator[](int i);  //!< access log elements, 0 is the oldest size() the newest
    bool hasNewMessages(); //!< has new messages since the last call?
    int size(); //!< number of entries in buffer

    // manage capacity
    void changeCapacity(int newCap); //!< set a new maximum size for the buffer
    int capacity(); //!< capacity of the buffer
    bool full(); //!< if buffer is full, old messages are overwritten
    bool empty(); //!< is buffer empty

private:
    std::vector<LogMessage> m_data;
    std::atomic_int m_insertLine; //!< the position where the next line will be written
    std::atomic_int m_readLine;  //!< the position of the oldest line
    std::shared_timed_mutex m_changeBufferMtx; //!< mutex to lock when changing the underlying buffer
    std::atomic_bool m_newMessage{false}; //!< has a new message?

    std::vector<std::reference_wrapper<LogMessage>> m_filtered;
    std::array<bool,7> m_allowedLogLvls{true,true,true,true,true,true,true}; //!< store which log levels should be displayed
    std::string m_moduleFilter; //!< filter module by string (include,-exclude)
    std::string m_messageFilter; //!< filter message by string (include,-exclude)
    std::thread::id m_tidFilter; //!< only display from this thread
    std::string m_fileFilter; //!< only display from this file

    std::atomic_bool m_newFilterState; //!< signal that the filter state was changed
    std::atomic_bool m_rebuildingTheFilterAsync; //!< currently rebuilding the filter
    std::atomic_bool m_pauseRebuilding; //!< pause rebuilding
    std::atomic_bool m_restartRebuilding; //!< restart rebuilding (eg after something else was also changed)
    void rebuildFilter(); //!< rebuilds the vector of filtered items
    bool checkFilter(const LogMessage& msg); //!< check msg should be displayed with current filter
};

//-------------------------------------------------------------------
/**
 * class BufferedSink
 *
 * usage:
 * Create a sink buffer and pass it to the buffer sink. You can then use the bufer to access logged data
 * or pass it to the Imgui logger window. The buffer must outlive the Sink
 *
 */
class BufferedSink
{
public:
    explicit BufferedSink(LogBuffer& buffer);
    BufferedSink(const BufferedSink& b)
        : m_buffer(b.m_buffer)
    {
    }
    void operator()(const LogMessage& msg);
private:
    LogBuffer& m_buffer;
};

}
#endif //MPUTILS_BUFFEREDSINK_H
