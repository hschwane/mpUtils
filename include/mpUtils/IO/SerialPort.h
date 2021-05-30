/*
 * mpUtils
 * SerialPort.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the SerialPort class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_SERIALPORT_H
#define MPUTILS_SERIALPORT_H

// includes
//--------------------
#ifdef _WIN32
    #error Serial port not implemented for windows
#elif __unix__
    #include "posixSerialPortDefines.h"
#else
    # error Serial port is not implemented for you Operating System!
#endif
#include <string>
#include <vector>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace io {
//--------------------

//-------------------------------------------------------------------
/**
 * class SerialPort
 *
 * Serial port class to connect to a serial port with basic setings.
 * Then you can perform blocking read and write operations on the serial port.
 *
 */
class SerialPort
{
public:

    //!< open a serial port with some sensible default settings
    explicit SerialPort(const std::string& fileName = "",
                        BaudRate baudRate = BaudRate::BAUD_9600,
                        CharSize characterSize = CharSize::CHAR_SIZE_8,
                        FlowControl flowControlType = FlowControl::FLOW_CONTROL_NONE,
                        Parity parityType = Parity::PARITY_NONE,
                        StopBits stopBits = StopBits::STOP_BITS_1);

    ~SerialPort(); //!< destructor closes file descriptor

    // make SerialPort non copyable but movable
    SerialPort(const SerialPort& other) = delete;
    SerialPort& operator=(const SerialPort& other) = delete;
    SerialPort(SerialPort&& other) noexcept : m_fd{} { *this = std::move(other); };
    SerialPort& operator=(SerialPort&& other) noexcept;

    void open(const std::string& fileName,
              BaudRate baudRate = BaudRate::BAUD_9600,
              CharSize characterSize = CharSize::CHAR_SIZE_8,
              FlowControl flowControlType = FlowControl::FLOW_CONTROL_NONE,
              Parity parityType = Parity::PARITY_NONE,
              StopBits stopBits = StopBits::STOP_BITS_1); //!< open the serial port by name
    void open(SerialDescriptor_t descriptor,
              BaudRate baudRate = BaudRate::BAUD_9600,
              CharSize characterSize = CharSize::CHAR_SIZE_8,
              FlowControl flowControlType = FlowControl::FLOW_CONTROL_NONE,
              Parity parityType = Parity::PARITY_NONE,
              StopBits stopBits = StopBits::STOP_BITS_1); //!< open port using existing descriptor
    void close(); //!< automatically called in destructor
    bool is_open() const; //!< check if serial port is open

    void setProperties(BaudRate baudRate = BaudRate::BAUD_9600,
                       CharSize characterSize = CharSize::CHAR_SIZE_8,
                       FlowControl flowControlType = FlowControl::FLOW_CONTROL_NONE,
                       Parity parityType = Parity::PARITY_NONE,
                       StopBits stopBits = StopBits::STOP_BITS_1); //!< sets serial port properties

    // set / get model lines
    bool getCTS() const;
    void setCTS(bool state);
    bool getRTS() const;
    void setRTS(bool state);
    bool getDTR() const;
    void setDTR(bool state);

    int charsAvailable() const; //!< number of bytes available for read
    void sync(); //!< blocks until all output in queue was send

    char readChar(); //!< block until character is read
    size_t read(char* buf, size_t count); //!< block until count elements are read
    void writeChar(char c); //!< blocks until c is passed to serial port driver
    void write(const char* buf, size_t count); //!< blocks until count elements from buf are passed to the serial port driver

    friend void swap(SerialPort& first, SerialPort& second); //!< swap two serial port instances

private:
    SerialDescriptor_t m_fd;
};

/**
 * @brief List all serial ports available on the system.
 * @return list of strings to all serial ports available on the system
 */
std::vector<std::string> listAvailableSerialPorts();

}}
#endif //MPUTILS_SERIALPORT_H
