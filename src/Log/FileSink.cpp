/*
 * mpUtils
 * FileSink.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the FileSink class
 *
 * Copyright (c) 2017 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Log/FileSink.h"
//--------------------

// namespace
//--------------------
namespace mpu{
//--------------------

// function definitions of the FileSink class
//-------------------------------------------------------------------
FileSink::FileSink(std::string sFilename, std:: size_t maxFileSize, int numLogsToKeep, bool printPlaintexts) : sLogfileName(sFilename), maxFileSize(maxFileSize), iNumLogsToKeep(numLogsToKeep), m_printPlaintexts(printPlaintexts)
{
    rotateLog();

    if (!file.is_open())
        throw std::runtime_error("Log: Could not open output file stream!");
}

void FileSink::operator()(const LogMessage &msg)
{
    if(maxFileSize != 0 && ((std::size_t)(file.tellp()) + msg.sMessage.size()) > maxFileSize)
    {
        rotateLog();
    }

    struct tm timeStruct;
#ifdef __linux__
    localtime_r(&msg.timepoint, &timeStruct);
#elif _WIN32
    localtime_s(&timeStruct, &msg.timepoint);
#else
#error please implement this for your operating system
#endif

    if(msg.plaintext)
    {
        if(m_printPlaintexts)
            file << msg.sMessage << std::endl;
    }
    else
    {
        file <<  "[" << toString(msg.lvl) << "]"
             << " [" << std::put_time( &timeStruct, "%c") << "]";

        if(!msg.sModule.empty())
            file << " (" << msg.sModule << "):";

        file << "\t" << msg.sMessage
             << "\tThread: " << std::setbase(16) << msg.threadId << std::setbase(10);

        if(!msg.sFilePosition.empty())
            file << "\t@File: " << msg.sFilePosition;

        file << std::endl;
    }
}

void FileSink::rotateLog()
{
    namespace fs = std::experimental::filesystem;
    if(file.is_open())
        file.close();

    // rename all existing files deleting the oldest (if logs kept is zero or one this will not be executed at all)
    for(int i=iNumLogsToKeep-1; i >= 1; i--)
    {
        if(fs::exists( sLogfileName + "." + std::to_string(i)))
            fs::rename( sLogfileName + "." + std::to_string(i), sLogfileName + "." + std::to_string(i+1));
    }

    // if we want to keep at least one, move the original
    if(iNumLogsToKeep > 0 && fs::exists( sLogfileName))
        fs::rename( sLogfileName, sLogfileName + ".1");

    file.open(sLogfileName, std::ofstream::out | std::ofstream::trunc);

    if (!file.is_open())
        throw std::runtime_error("Log: Could not open output file stream!");
}


}