/*
 * mpUtils
 * SerialPortStreambuf.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the SerialPortStreambuf class
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/IO/SerialPortStream.h"
//--------------------

// namespace
//--------------------
namespace mpu{
namespace io {
//--------------------

// function definitions of the SerialPortStreambuf class
//-------------------------------------------------------------------

SerialPortStreambuf::SerialPortStreambuf(const std::string& fileName, BaudRate baudRate,
                                         CharSize characterSize, FlowControl flowControlType,
                                         Parity parityType, StopBits stopBits)
    : m_serial(fileName,baudRate,characterSize,flowControlType,parityType,stopBits), m_putback()
{
    setg(nullptr, nullptr, nullptr);
}

void SerialPortStreambuf::open(const std::string& fileName, BaudRate baudRate, CharSize characterSize,
                               FlowControl flowControlType, Parity parityType, StopBits stopBits)
{
    m_serial.open(fileName,baudRate,characterSize,flowControlType,parityType,stopBits);
}

void SerialPortStreambuf::close()
{
    m_serial.close();
}

bool SerialPortStreambuf::is_open() const
{
    return m_serial.is_open();
}

bool SerialPortStreambuf::getCTS() const
{
    return m_serial.getCTS();
}

void SerialPortStreambuf::setCTS(bool state)
{
    m_serial.setCTS(state);
}

bool SerialPortStreambuf::getRTS() const
{
    return m_serial.getRTS();
}

void SerialPortStreambuf::setRTS(bool state)
{
    m_serial.setRTS(state);
}

bool SerialPortStreambuf::getDTR() const
{
    return m_serial.getDTR();
}

void SerialPortStreambuf::setDTR(bool state)
{
    m_serial.setDTR(state);
}

SerialPortStreambuf::int_type SerialPortStreambuf::sync()
{
    m_serial.sync();
    return traits_type::not_eof(0);
}

std::streamsize SerialPortStreambuf::showmanyc()
{
    return m_serial.charsAvailable() + m_putback.size();
}

std::streamsize SerialPortStreambuf::xsgetn(char_type* __s, std::streamsize __n)
{
    setg(nullptr, nullptr, nullptr);

    int num=0;
    while(!m_putback.empty() && __n > 0) {
        *__s = m_putback.back();
        m_putback.pop_back();
        ++__s;
        --__n;
        ++num;
    }

    if(__n > 0)
        num += m_serial.read(__s, __n);

    return num;
}

SerialPortStreambuf::int_type SerialPortStreambuf::uflow()
{
    // remove from putback or load, also advance
    char_type c;
    if(m_putback.empty()) {
        c = m_serial.readChar();
    } else {
        c = m_putback.back();
        m_putback.pop_back();
    }

    return traits_type::to_int_type(c);
}

SerialPortStreambuf::int_type SerialPortStreambuf::underflow()
{
    // read character without advancing (i.e. leave it in putback)
    char_type c;
    if(m_putback.empty()) {
        c = m_serial.readChar();
        m_putback.push_back(c);
    } else {
        c = m_putback.back();
    }
    return traits_type::to_int_type(c);
}

SerialPortStreambuf::int_type SerialPortStreambuf::pbackfail(int_type __c)
{
    if (traits_type::eq_int_type(__c, traits_type::eof()))
        return traits_type::eof() ;

    m_putback.push_back(traits_type::to_char_type(__c));
    return traits_type::not_eof(__c) ;
}

std::streamsize SerialPortStreambuf::xsputn(const char_type* __s, std::streamsize __n)
{
    m_serial.write(__s,__n);
    return __n;
}

SerialPortStreambuf::int_type SerialPortStreambuf::overflow(int_type __c)
{
    if (traits_type::eq_int_type(__c, traits_type::eof()))
        return traits_type::eof() ;

    m_serial.writeChar(traits_type::to_char_type(__c));
    return traits_type::not_eof(__c);
}

void swap(SerialPortStreambuf& first, SerialPortStreambuf& second)
{
    using std::swap;
    swap(first.m_serial,second.m_serial);
}

SerialPortStream::SerialPortStream(const std::string& fileName, BaudRate baudRate, CharSize characterSize,
                                   FlowControl flowControlType, Parity parityType, StopBits stopBits)
    : m_streambuf(fileName,baudRate,characterSize,flowControlType,parityType,stopBits), std::iostream(&m_streambuf)
{
}

void SerialPortStream::open(const std::string& fileName, BaudRate baudRate, CharSize characterSize,
                            FlowControl flowControlType, Parity parityType, StopBits stopBits)
{
    m_streambuf.open(fileName,baudRate,characterSize,flowControlType,parityType,stopBits);
}

void SerialPortStream::close()
{
    m_streambuf.close();
}

bool SerialPortStream::is_open() const
{
    return m_streambuf.is_open();
}

bool SerialPortStream::getCTS() const
{
    return m_streambuf.getCTS();
}

void SerialPortStream::setCTS(bool state)
{
    m_streambuf.setCTS(state);
}

bool SerialPortStream::getRTS() const
{
    return m_streambuf.getRTS();
}

void SerialPortStream::setRTS(bool state)
{
    m_streambuf.setRTS(state);
}

bool SerialPortStream::getDTR() const
{
    return m_streambuf.getDTR();
}

void SerialPortStream::setDTR(bool state)
{
    m_streambuf.setDTR(state);
}

SerialPortStreambuf* SerialPortStream::rdbuf()
{
    return &m_streambuf;
}

const SerialPortStreambuf* SerialPortStream::rdbuf() const
{
    return &m_streambuf;
}

int SerialPortStream::charsAvailable()
{
    return m_streambuf.in_avail();
}

void swap(SerialPortStream& first, SerialPortStream& second)
{
    using std::swap;
    swap(first.m_streambuf,second.m_streambuf);
}



}}