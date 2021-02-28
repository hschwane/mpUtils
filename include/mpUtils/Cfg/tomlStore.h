/*
 * mpUtils
 * tomlStore.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_TOMLSTORE_H
#define MPUTILS_TOMLSTORE_H

// includes
//--------------------
#include "mpUtils/external/toml/toml.hpp"
#include "mpUtils/Log/Log.h"
//--------------------

namespace toml {

//!< stores toml table in file
inline void store(const std::string& filename, const toml::value& tlv)
{
    std::ofstream outFile(filename, std::fstream::out | std::fstream::trunc);
    assert_critical(outFile.is_open(),"tomlStore", "File " + filename + "could not be opened for writing");
    outFile << std::setw(120) << tlv;
}

//!< appends toml table to file
//!< no checks are performed to make sure the resulting file is a proper toml file
inline void storeAppend(const std::string& filename, const toml::value& tlv)
{
    std::ofstream outFile(filename, std::fstream::out | std::fstream::app);
    assert_critical(outFile.is_open(),"tomlStore", "File " + filename + "could not be opened for writing");
    outFile << std::setw(120) << tlv;
}

}

#endif //MPUTILS_TOMLSTORE_H
