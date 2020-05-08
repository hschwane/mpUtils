/*
 * mpUtils
 * mpLog.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Log class, which provides logging to text files, syslog, or custom streams
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include <mpUtils/Log/Log.h>
#include "mpUtils/version.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

// Variables for logToString
const std::string LogLvlToString[] = {"NOLOG", "FATAL_ERROR", "ERROR", "WARNING", "INFO", "DEBUG", "DEBUG2",
                                      "ALL"};
const std::string LogLvlStringInvalid = "INVALID";

// functions of the Log class
//-------------------------------------------------------------------
Log::~Log()
{
    close();
    if(globalLog == this)
        globalLog = nullptr;
}

void Log::removeSink(int index)
{
    std::lock_guard<std::mutex> lck(loggerMtx);
    printFunctions.erase( printFunctions.begin() + index);
}

void Log::close()
{
    // accept no more messages
    LogLvl oldLvl = logLvl;
    logLvl = LogLvl::NOLOG;

    // flush it
    std::unique_lock<std::mutex> lck(loggerMtx);
    if(bShouldLoggerRun)
    {
        bShouldLoggerRun = false;
        loggerCv.notify_one();
    }
    lck.unlock();
    if(loggerMainThread.joinable())
        loggerMainThread.join();
    lck.lock();

    // remove all sinks
    printFunctions.clear();
    logLvl = oldLvl;
}

void Log::flush()
{
    std::unique_lock<std::mutex> lck(loggerMtx);

    // accept no more messages
    LogLvl oldLvl = logLvl;
    logLvl = LogLvl::NOLOG;

    // wait for the logger to print all queued messages and join the thread
    if(bShouldLoggerRun)
    {
        bShouldLoggerRun = false;
        loggerCv.notify_one();
    }
    lck.unlock();
    if(loggerMainThread.joinable())
        loggerMainThread.join();
    lck.lock();

    // restart the logger
    logLvl = oldLvl;
    if(!bShouldLoggerRun)
    {
        bShouldLoggerRun = true;
        loggerMainThread = std::thread(&Log::loggerMainfunc, this);
    }
}

LogStream Log::print(const LogLvl lvl)
{
    LogMessage* lm = new LogMessage;
    lm->lvl = lvl;
    lm->plaintext=true;
    return LogStream( (*this), lm);
}

void Log::printHeader(std::string appName, std::string appVersion, std::string appSha, std::string appBuildType)
{
    // use error log level so this is written almost always
    print(LogLvl::INFO)     << "*===================================================\n"
                            << "*             -- " << appName << " --\n"
                            << "*          version: " << appVersion << "\n"
                            << "*          commit: " << appSha << "\n"
                            << "*          buil-type: " << appBuildType << "\n"
                            << "* using mpUtils version " << MPU_VERSION_STRING << " " << MPU_VERSION_COMMIT
#if defined(NDEBUG)
                            << " Release\n"
#else
                            << " Debug\n"
#endif
                            << "*===================================================";
}

void Log::logMessage(LogMessage* lm)
{
    if(!printFunctions.empty() && lm->lvl <= logLvl)
    {
        std::lock_guard<std::mutex> lck(queueMtx);
        messageQueue.push(lm);
        loggerCv.notify_one();
    }
}

LogStream Log::operator()(const LogLvl lvl, std::string&& sFilepos, std::string&& sModule)
{
    LogMessage* lm = new LogMessage;
    lm->lvl = lvl;
    lm->sFilePosition = std::move(sFilepos);
    lm->sModule = std::move(sModule);
    lm->threadId = std::this_thread::get_id();
    lm->timepoint = time(nullptr);

    return LogStream( (*this), lm);
}

void Log::loggerMainfunc()
{
    std::unique_lock<std::mutex> lck(loggerMtx);
    do
    {
        loggerCv.wait(lck);

        std::unique_lock<std::mutex> queueLck(queueMtx);
        while(!messageQueue.empty())
        {
            auto msg= messageQueue.front();
            messageQueue.pop();
            queueLck.unlock();

            // print to all sinks
            for(auto &&function : printFunctions)
            {
                function(*msg);
            }

            delete(msg);

            queueLck.lock();
        }
    }
    while(bShouldLoggerRun);
}

// static variables
Log* Log::globalLog = nullptr;

}