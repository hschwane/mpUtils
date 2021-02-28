/*
 * mpUtils
 * posixSerialPort.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */


// includes
//--------------------
#include "mpUtils/IO/SerialPort.h"
#include "mpUtils/Misc/timeUtils.h"
#include "mpUtils/Log/Log.h"

#include <iostream>
#include <stdexcept>
#include <filesystem>

#include <cstdio>   /* Standard input/output definitions */
#include <cstring>  /* String function definitions */
#include <cerrno>   /* Error number definitions */

#include <unistd.h>  /* UNIX standard function definitions */
#include <fcntl.h>   /* File control definitions */
#include <sys/ioctl.h>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace io {
namespace fs = std::filesystem;
//--------------------

// function definitions of the SerialPort class
//-------------------------------------------------------------------
SerialPort::SerialPort(const std::string& fileName, BaudRate baudRate, CharSize characterSize,
                       FlowControl flowControlType, Parity parityType, StopBits stopBits)
        : m_fd(-1)
{
    if(!fileName.empty())
        this->open(fileName, baudRate, characterSize, flowControlType, parityType, stopBits);
}

SerialPort::~SerialPort()
{
    if(is_open())
        close();
}

SerialPort& SerialPort::operator=(SerialPort&& other) noexcept
{
    using std::swap;
    swap(other.m_fd, m_fd);
    return *this;
}

void SerialPort::open(const std::string& fileName, BaudRate baudRate, CharSize characterSize,
                      FlowControl flowControlType, Parity parityType, StopBits stopBits)
{
    if(is_open())
        throw std::logic_error("Called open(), but serial port is already open.");

    m_fd = ::open(fileName.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if(m_fd < 0)
        throw std::runtime_error("Could not open serial \"" + fileName + "\": " + std::strerror(errno));

    if(::ioctl(m_fd, TIOCEXCL) < 0)
        throw std::runtime_error("Could block serial port \"" + fileName + "\": " + std::strerror(errno));

    // settings
    termios settings{};

    // set baud rate
    cfsetspeed(&settings, static_cast<speed_t>(baudRate));
    cfsetospeed(&settings, static_cast<speed_t>(baudRate));

    // set the character size
    if(characterSize == CharSize::CHAR_SIZE_8)
        settings.c_iflag &= ~ISTRIP;
    else
        settings.c_iflag |= ISTRIP;
    settings.c_cflag &= ~CSIZE;                               // clear all CSIZE bits.
    settings.c_cflag |= static_cast<tcflag_t>(characterSize); // set the character size.

    // set the flow control setting
    switch(flowControlType) {
        case FlowControl::FLOW_CONTROL_HARDWARE:
            settings.c_iflag &= ~(IXON | IXOFF | IXANY);
            settings.c_cflag |= CRTSCTS;
            settings.c_cc[VSTART] = _POSIX_VDISABLE;
            settings.c_cc[VSTOP] = _POSIX_VDISABLE;
            break;
        case FlowControl::FLOW_CONTROL_SOFTWARE:
            settings.c_iflag |= IXON | IXOFF | IXANY;
            settings.c_cflag &= ~CRTSCTS;
            settings.c_cc[VSTART] = 021;
            settings.c_cc[VSTOP] = 023;
            break;
        case FlowControl::FLOW_CONTROL_NONE:
            settings.c_iflag &= ~(IXON | IXOFF);
            settings.c_cflag &= ~CRTSCTS;
            break;
        default:
            throw std::invalid_argument("invalid flow control setting");
    }

    // set parity
    switch(parityType) {
        case Parity::PARITY_EVEN:
            settings.c_cflag |= PARENB;
            settings.c_cflag &= ~PARODD;
            settings.c_iflag |= INPCK | ISTRIP;
            break;
        case Parity::PARITY_ODD:
            settings.c_cflag |= PARENB;
            settings.c_cflag |= PARODD;
            settings.c_iflag |= INPCK | ISTRIP;
            break;
        case Parity::PARITY_NONE:
            settings.c_cflag &= ~PARENB;
            settings.c_iflag |= IGNPAR;
            break;
        default:
            throw std::invalid_argument("invalid parity setting");
    }

    // set stop bit
    switch(stopBits) {
        case StopBits::STOP_BITS_1:
            settings.c_cflag &= ~CSTOPB;
            break;
        case StopBits::STOP_BITS_2:
            settings.c_cflag |= CSTOPB;
            break;
        default:
            throw std::invalid_argument("invalid stop bit setting");
    }

    // enable read and ignore model control
    settings.c_cflag |= CLOCAL | CREAD;

    // ignore breaks
    settings.c_iflag |= IGNBRK;

    // unprocessed output
    settings.c_oflag &= ~OPOST;

    if(tcsetattr(m_fd, TCSANOW, &settings) < 0)
        throw std::runtime_error("Could read serial port \"" + fileName + "\" settings: " + std::strerror(errno));
}

void SerialPort::close()
{
    if(!is_open())
        throw std::logic_error("Called close(), but serial port is not open.");
    ::tcdrain(m_fd);
    ::close(m_fd);
    m_fd = -1;
}

bool SerialPort::is_open() const
{
    return (m_fd >= 0);
}

bool SerialPort::getCTS() const
{
    int state = -1;
    if(::ioctl(m_fd, TIOCMGET, &state) < 0)
        throw std::runtime_error(std::strerror(errno));
    return (0 != (state & TIOCM_CTS));
}

void SerialPort::setCTS(bool state)
{
    if(::ioctl(m_fd, state ? TIOCMBIS : TIOCMBIC, TIOCM_CTS) < 0)
        throw std::runtime_error(std::strerror(errno));
}

bool SerialPort::getRTS() const
{
    int state = -1;
    if(::ioctl(m_fd, TIOCMGET, &state) < 0)
        throw std::runtime_error(std::strerror(errno));
    return (0 != (state & TIOCM_RTS));
}

void SerialPort::setRTS(bool state)
{
    if(::ioctl(m_fd, state ? TIOCMBIS : TIOCMBIC, TIOCM_RTS) < 0)
        throw std::runtime_error(std::strerror(errno));
}

bool SerialPort::getDTR() const
{
    int state = -1;
    if(::ioctl(m_fd, TIOCMGET, &state) < 0)
        throw std::runtime_error(std::strerror(errno));
    return (0 != (state & TIOCM_DTR));
}

void SerialPort::setDTR(bool state)
{
    if(::ioctl(m_fd, state ? TIOCMBIS : TIOCMBIC, TIOCM_DTR) < 0)
        throw std::runtime_error(std::strerror(errno));
}

int SerialPort::bytesAvailable() const
{
    int bytes = 0;
    if(::ioctl(m_fd, FIONREAD, &bytes) < 0)
        throw std::runtime_error(std::strerror(errno));
    return bytes;
}

void SerialPort::sync()
{
    if(::tcdrain(m_fd) < 0)
        throw std::runtime_error(std::strerror(errno));
}

char SerialPort::readChar()
{
    char c;
    int r = -1;
    while(true) {
        r = ::read(m_fd, &c, 1);
        if(r>0)
            break;
        else if(r<0 && errno != EAGAIN && errno != EWOULDBLOCK)
            throw std::runtime_error(std::string("Error reading from serial: ") + std::strerror(errno));
        sleep_ms(2);
    }
    return c;
}

size_t SerialPort::read(char* buf, size_t count)
{
    std::cout << "blubber" << std::endl;

    int r = -1;
    size_t rec = 0;
    while(true) {
        r = ::read(m_fd, buf+rec, count-rec);
        if(r>0)
            rec += r;
        else if(r<0 && errno != EAGAIN && errno != EWOULDBLOCK)
            throw std::runtime_error(std::string("Error reading from serial: ") + std::strerror(errno));
        if(rec>=count)
            break;
        sleep_ms(2);
    }
    return rec;
}

void SerialPort::writeChar(char c)
{
    int r = -1;
    while(true) {
        r = ::write(m_fd, &c, 1);
        if(r>0)
            break;
        else if(r<0 && errno != EAGAIN && errno != EWOULDBLOCK)
            throw std::runtime_error(std::string("Error writing to serial: ") + ::strerror(errno));
        sleep_ms(2);
    }
}

void SerialPort::write(const char* buf, size_t count)
{
    int r = -1;
    size_t send = 0;
    while(true) {
        r = ::write(m_fd, buf+send, count-send);
        if(r>0)
            send += r;
        else if(r<0 && errno != EAGAIN && errno != EWOULDBLOCK)
            throw std::runtime_error(std::string("Error writing to serial: ") + std::strerror(errno));
        if(send>=count)
            break;
        sleep_ms(2);
    }
}

void swap(SerialPort& first, SerialPort& second)
{
    using std::swap;
    swap(first.m_fd,second.m_fd);
}

std::vector<std::string> listAvailableSerialPorts()
{
    std::vector<std::string> port_names;

    fs::path p("/dev/serial/by-path");
    try {
        if (!exists(p)) {
            throw std::runtime_error(p.generic_string() + " does not exist");
        } else {
            for (const fs::directory_entry &de : fs::directory_iterator(p)) {
                if (fs::is_symlink(de)) {
                    fs::path linkTarget = fs::read_symlink(de);
                    fs::path path;
                    if(linkTarget.is_relative()) {
                        path = p;
                        path /= linkTarget;
                    } else {
                        path = linkTarget;
                    }
                    port_names.push_back(fs::canonical(path).string());
                }
            }
        }
    } catch (const fs::filesystem_error &ex) {
        logERROR("SerialPort") << "Failed listing serial ports: " << ex.what();
        throw ex;
    }
    std::sort(port_names.begin(), port_names.end());
    return port_names;
}

}}