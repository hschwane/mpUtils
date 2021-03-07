/*
 * mpUtils
 * SerialPortStreambuf.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the SerialPortStreambuf class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_SERIALPORTSTREAM_H
#define MPUTILS_SERIALPORTSTREAM_H

// includes
//--------------------
#include "SerialPort.h"
#include <streambuf>
#include <iostream>
#include <vector>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace io {
//--------------------

//-------------------------------------------------------------------
/**
 * class SerialPortStreambuf
 *
 * usage:
 * Used as a buffer for an std::iostream. Or better use the SerialPortStream below.
 *
 */
class SerialPortStreambuf : public std::streambuf
{
public:
    using char_type = typename std::streambuf::char_type;
    using int_type = typename std::streambuf::int_type;

    //!< open a serial port with some sensible default settings
    explicit SerialPortStreambuf(const std::string& fileName = "",
                              BaudRate baudRate = BaudRate::BAUD_9600,
                              CharSize characterSize = CharSize::CHAR_SIZE_8,
                              FlowControl flowControlType = FlowControl::FLOW_CONTROL_NONE,
                              Parity parityType = Parity::PARITY_NONE,
                              StopBits stopBits = StopBits::STOP_BITS_1);

    //!< open a serial port with some sensible default settings
    void open(const std::string& fileName,
              BaudRate baudRate = BaudRate::BAUD_9600,
              CharSize characterSize = CharSize::CHAR_SIZE_8,
              FlowControl flowControlType = FlowControl::FLOW_CONTROL_NONE,
              Parity parityType = Parity::PARITY_NONE,
              StopBits stopBits = StopBits::STOP_BITS_1);
    void close(); //!< automatically called in destructor
    bool is_open() const; //!< check if serial port is open

    // set / get modem lines
    bool getCTS() const; //!< get state of CTS line
    void setCTS(bool state); //!< set state of CTS line
    bool getRTS() const; //!< get state of RTS line
    void setRTS(bool state); //!< set state of RTS line
    bool getDTR() const; //!< get state of DTR line
    void setDTR(bool state); //!< set state of DTR line

    friend void swap(SerialPortStreambuf& first, SerialPortStreambuf& second); //!< swap two serial port streambufs

protected:
    // overrides from streambuf
    SerialPortStreambuf::int_type sync() override;
    std::streamsize showmanyc() override;
    std::streamsize xsgetn(char_type* __s, std::streamsize __n) override;
    SerialPortStreambuf::int_type uflow() override;
    SerialPortStreambuf::int_type underflow() override;
    SerialPortStreambuf::int_type pbackfail(int_type __c) override;
    std::streamsize xsputn(const char_type* __s, std::streamsize __n) override;
    SerialPortStreambuf::int_type overflow(int_type __c) override;

private:
    // internal data
    SerialPort m_serial;
    std::vector<char_type> m_putback;
};

//-------------------------------------------------------------------
/**
 * class SerialPortStream
 *
 * Stream to interact with serial ports. Use like an iostream. Pass serial port settings to the constructor.
 * You can use get / set functions to manually control RTS / CTS and DTR lines.
 * Use listAvailableSerialPorts() to list all serial ports available on the system.
 *
 */
class SerialPortStream : public std::iostream
{
public:
    //!< open a serial port with some sensible default settings
    explicit SerialPortStream(const std::string& fileName = "",
                                 BaudRate baudRate = BaudRate::BAUD_9600,
                                 CharSize characterSize = CharSize::CHAR_SIZE_8,
                                 FlowControl flowControlType = FlowControl::FLOW_CONTROL_NONE,
                                 Parity parityType = Parity::PARITY_NONE,
                                 StopBits stopBits = StopBits::STOP_BITS_1);

    //!< open a serial port with some sensible default settings
    void open(const std::string& fileName,
              BaudRate baudRate = BaudRate::BAUD_9600,
              CharSize characterSize = CharSize::CHAR_SIZE_8,
              FlowControl flowControlType = FlowControl::FLOW_CONTROL_NONE,
              Parity parityType = Parity::PARITY_NONE,
              StopBits stopBits = StopBits::STOP_BITS_1);
    void close(); //!< automatically closed in destructor
    bool is_open() const; //!< check if serial port is open

    int charsAvailable(); //!< number of characters available for reading

    // set / get modem lines
    bool getCTS() const; //!< get state of CTS line
    void setCTS(bool state); //!< set state of CTS line
    bool getRTS() const; //!< get state of RTS line
    void setRTS(bool state); //!< set state of RTS line
    bool getDTR() const; //!< get state of DTR line
    void setDTR(bool state); //!< set state of DTR line

    SerialPortStreambuf* rdbuf(); //!< return pointer to internal serial port streambuf
    const SerialPortStreambuf* rdbuf() const; //!< return pointer to internal serial port streambuf
    friend void swap(SerialPortStream& first, SerialPortStream& second); //!< swap two serial port streambufs

public:
    SerialPortStreambuf m_streambuf;
};

}}
#endif //MPUTILS_SERIALPORTSTREAM_H
