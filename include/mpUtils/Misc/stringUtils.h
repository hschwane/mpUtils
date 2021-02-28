/*
 * mpUtils
 * mpUtils.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Defines some basic string functions.
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

#ifndef MPUTILS_STRINGUTILS_H
#define MPUTILS_STRINGUTILS_H

// includes
//--------------------
#include <string>
#include <ctime>
#include <algorithm>
#include <sstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <string_view>
#include <vector>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------
//--------------------
// some string helper functions

std::string timestamp(const std::string& sFormat = "%c"); //!<  get current timestamp as string

// useful character collections
constexpr char whitespace[] = " \f\n\r\t\v"; //!< string of witespace chars

// dealing with escaped characters
std::string &escapeString(std::string &s, std::string sToEscape, char cEscapeChar = '\\'); //!< escapes all chars from sToEscape in s using cEscapeChar
std::string &unescapeString(std::string &s, char cEscapeChar = '\\'); //!< removes all cEscapeChars from the string but allow the escapeChar
size_t findFirstNotEscapedOf(const std::string &s, const std::string &c, size_t pos = 0, const std::string &sEscape = "\\"); //!< returns the position of the first char from c in s after pos which is not escaped by a char from sEscape
std::string &cutAfterFirst(std::string &s, const std::string &c, const std::string &sEscape = "", size_t pos = 0); //!< cuts the first found char in c after pos and everything after that from s stuff can be escaped by any of the chars in sEscape

/**
 * @brief tokenize string at any char out of delims, removing the delimiter
 * @param s string to tokenize
 * @param delims list of delimiters
 * @return vector of tokens
 */
std::vector<std::string_view> tokenize(std::string_view s, std::string_view delims=whitespace);

/**
 * @brief tokenize string at delim, removing the delimiter
 * @param s string to tokenize
 * @param delim the delimiter
 * @return vector of tokens
 */
std::vector<std::string_view> tokenize(std::string_view s, char delim);

}
#endif //MPUTILS_MPUTILS_H
