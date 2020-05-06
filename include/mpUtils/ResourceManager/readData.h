/*
 * mpUtils
 * readData.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_READDATA_H
#define MPUTILS_READDATA_H

// includes
//--------------------
#include <string>
#include <fstream>
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
// http://insanecoding.blogspot.com/2011/11/how-to-read-in-file-in-c.html
/**
 * @brief Loads data from file into string. Works with binary and text files.
 * @param path path to the file
 * @return string containing data of the file
 */
std::string readFile(const std::string& path)
{
    if(*(path.end()-1) == '/')
        throw std::runtime_error("Passed path to folder instead of file: " + path);

    std::ifstream in(path, std::ios::in | std::ios::binary);

    if(!in.is_open())
        throw std::runtime_error("Could not open file " + path);

    std::string contents;
    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return (contents);
}

}
#endif //MPUTILS_READDATA_H
