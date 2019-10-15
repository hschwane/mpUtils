/*
 * mpUtils
 * ConsoleSink.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the ConsoleSink class
 *
 * Copyright (c) 2017 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Log/ConsoleSink.h"
#include <iostream>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

// function definitions of the ConsoleSink class
//-------------------------------------------------------------------
void ConsoleSink::operator()(const LogMessage &msg)
{
    std::ostream* os = &std::cout;

    struct tm timeStruct;
#ifdef __linux__
    localtime_r(&msg.timepoint, &timeStruct);
#elif _WIN32
    localtime_s(&timeStruct, &msg.timepoint);
#else
#error "please implement this for your operating system"
#endif

    if(msg.plaintext)
    {
        *os << msg.sMessage << std::endl;
    }
    else
    {
        *os << "\033[1;" << levelToColor(msg.lvl) << "m"
            << "[" << toString(msg.lvl) << "]" << "\33[1;90m"
            << " [" << std::put_time(&timeStruct, "%x %X") << "]"
            << "\033[m ";

        if(!msg.sModue.empty())
            *os << " (" << msg.sModue << "):";

        *os << "\t" << msg.sMessage << "\33[1;90m"
            << "\tThread: " << std::setbase(16) << msg.threadId
            << std::setbase(10)
            << "\033[m" ;

        if(!msg.sFilePosition.empty())
            *os << "\33[1;90m"
                << "\t@File: " << msg.sFilePosition << "\033[m";

        *os << std::endl;
    }
}

constexpr int ConsoleSink::levelToColor(LogLvl lvl)
{
    switch(lvl)
    {
        case LogLvl::FATAL_ERROR:
        case LogLvl::ERROR:
            return 31;
        case LogLvl::WARNING:
            return 33;
        case LogLvl::INFO:
            return 32;
        case LogLvl::DEBUG:
        case LogLvl::DEBUG2:
            return 35;
        default:
            return 38;
    }
}

}
