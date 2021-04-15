/*
 * mpUtils
 * posixSerialPortDefines.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_POSIXSERIALPORTDEFINES_H
#define MPUTILS_POSIXSERIALPORTDEFINES_H

// includes
//--------------------
#include <limits>
#include <termios.h> /* POSIX terminal control definitions */
//--------------------

// namespace
//--------------------
namespace mpu {
namespace io {
//--------------------

/**
 * @brief enum of supported baud rates
 */
enum class BaudRate : speed_t
{
    BAUD_50 = B50,
    BAUD_75 = B75,
    BAUD_110 = B110,
    BAUD_134 = B134,
    BAUD_150 = B150,
    BAUD_200 = B200,
    BAUD_300 = B300,
    BAUD_600 = B600,
    BAUD_1200 = B1200,
    BAUD_1800 = B1800,
    BAUD_2400 = B2400,
    BAUD_4800 = B4800,
    BAUD_9600 = B9600,
    BAUD_19200 = B19200,
    BAUD_38400 = B38400,
    BAUD_57600 = B57600,
    BAUD_115200 = B115200,
    BAUD_230400 = B230400,
// some baud rates only exist for linux not other posix
#ifdef __linux__
    BAUD_460800 = B460800,
    BAUD_500000 = B500000,
    BAUD_576000 = B576000,
    BAUD_921600 = B921600,
    BAUD_1000000 = B1000000,
    BAUD_1152000 = B1152000,
    BAUD_1500000 = B1500000,
#if __MAX_BAUD > B2000000
    BAUD_2000000 = B2000000,
    BAUD_2500000 = B2500000,
    BAUD_3000000 = B3000000,
    BAUD_3500000 = B3500000,
    BAUD_4000000 = B4000000,
#endif // __MAX_BAUD
#endif // __linux__
};

/**
 * @brief list of allowed character sizes
 */
enum class CharSize : tcflag_t
{
    CHAR_SIZE_5 = CS5,
    CHAR_SIZE_6 = CS6,
    CHAR_SIZE_7 = CS7,
    CHAR_SIZE_8 = CS8,
};

/**
 * @brief list of allowed flow control types
 */
enum class FlowControl
{
    FLOW_CONTROL_HARDWARE,
    FLOW_CONTROL_SOFTWARE,
    FLOW_CONTROL_NONE,
};

/**
 * @brief list of allowed parity types
 */
enum class Parity
{
    PARITY_EVEN,
    PARITY_ODD,
    PARITY_NONE,
};

/**
 * @brief list of allowed number of stop bits
 */
enum class StopBits
{
    STOP_BITS_1,
    STOP_BITS_2,
};

//!< type of the serial port descriptor
using SerialDescriptor_t = int;

}}

#endif //MPUTILS_POSIXSERIALPORTDEFINES_H
