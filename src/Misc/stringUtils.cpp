/*
 * mpUtils
 * mpUtils.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements some basic string functions.
 *
 * Copyright 2016 Hendrik Schwanekamp
 */

// includes
//--------------------
#include "mpUtils/Misc/stringUtils.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

// general help functions
//-------------------------------------------------------------------

std::string timestamp(const std::string& sFormat)
{
    time_t timeRaw = time(nullptr);
    struct tm timeStruct{};
#ifdef __linux__
	localtime_r(&timeRaw, &timeStruct);
#elif _WIN32
	localtime_s(&timeStruct, &timeRaw);
#else
	#error please implement timestamp for your operating system
#endif
	const char* cstr = sFormat.c_str();
	std::ostringstream ss;
	ss << std::put_time(&timeStruct, cstr);
    return ss.str();
}

std::string &cutAfterFirst(std::string &s, const std::string &c, const std::string &sEscape, std::size_t pos)
{
    std::string sSearch = c + sEscape;
    size_t cut = s.find_first_of(sSearch, pos);
    while (cut != std::string::npos)
    {
        if (c.find_first_of(s[cut]) != std::string::npos) // check if this is a cutting char
        {
            s.erase(cut);
            break;
        }
        // if not it is the escape char, so we ignore the next char
        cut = s.find_first_of(sSearch, cut + 2);
    }

    return s;
}

std::size_t findFirstNotEscapedOf(const std::string &s, const std::string &c, std::size_t pos, const std::string &sEscape)
{
    std::string sSearch = c + sEscape;
    size_t search = s.find_first_of(sSearch, pos);
    while (search != std::string::npos)
    {
        if (c.find_first_of(s[search]) != std::string::npos) // check if this is a cutting char
            return search;
        // if not it is the escape char, so we ignore the next char
        search = s.find_first_of(sSearch, search + 2);
    }

    return std::string::npos;
}

std::string &escapeString(std::string &s, std::string sToEscape, const char cEscapeChar)
{
    sToEscape.push_back(cEscapeChar);
    size_t pos = s.find_first_of(sToEscape);

    while (pos != std::string::npos)
    {
        s.insert(pos, 1, cEscapeChar);
        pos = s.find_first_of(sToEscape, pos + 2);
    }

    return s;
}

std::string &unescapeString(std::string &s, const char cEscapeChar)
{
    size_t pos = s.find(cEscapeChar);
    while (pos != std::string::npos)
    {
        s.erase(pos, 1);
        pos = s.find(cEscapeChar, pos + 1);
    }

    return s;
}

std::vector<std::string_view> tokenize(std::string_view s, std::string_view delims)
{
    std::vector<std::string_view> tokens;
    auto p1 = s.find_first_not_of(delims);
    std::string_view::size_type p2;

    while((p2 = s.find_first_of(delims,p1)) != std::string_view::npos) {
        tokens.emplace_back(s.substr(p1, p2-p1));
        p1 = s.find_first_not_of(delims,p2+1);
    }
    if(p1 != std::string_view::npos)
        tokens.emplace_back(s.substr(p1));
    return tokens;
}

std::vector<std::string_view> tokenize(std::string_view s, char delim)
{
    std::vector<std::string_view> tokens;
    auto p1 = s.find_first_not_of(delim);
    std::string_view::size_type p2;

    while((p2 = s.find_first_of(delim,p1)) != std::string_view::npos) {
        tokens.emplace_back(s.substr(p1, p2-p1));
        p1 = s.find_first_not_of(delim,p2+1);
    }

    if(p1 != std::string_view::npos)
        tokens.emplace_back(s.substr(p1));
    return tokens;
}

}