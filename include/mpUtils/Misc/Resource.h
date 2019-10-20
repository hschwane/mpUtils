/*
 * mpUtils
 * Resource.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Resource class
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 * Based on blogpost: https://beesbuzz.biz/code/4399-Embedding-binary-resources-with-CMake-and-C-11 (found on 08.04.2019)
 *
 */

#ifndef MPUTILS_RESOURCE_H
#define MPUTILS_RESOURCE_H

// includes
//--------------------
#include <cstddef>
#include "mpUtils/external/incbin/incbin.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
/**
 * class Resource
 * Access a resource embedded into the executable.
 *
 * usage:
 * Use the ADD_RESOURCES macro to embed a resource file into a executable.
 * You can set the search path by using the compiler option -Wa,-I/some/path in your cmake
 * Then use the macro below to load the resource. It will return an object of the following class.
 * You can then use the iterators as well as data and size member functions to access the data.
 *
 * example:
 * int main(void)
 * {
 *   Resource text = LOAD_RESOURCE(resources_test_txt);
 *   std::cout << std::string(text.signedData(), text.size()) << std::endl;
 * }
 * where the file was located at resources/test.txt
 *
 */
class Resource
{
public:
    Resource(const unsigned char *start, const unsigned char *end) : m_data(start), m_signedData(reinterpret_cast<const char*>(start)),
                                                                    mSize(end - start)
    {}

    Resource(const unsigned char *start, size_t size) : m_data(start), m_signedData(reinterpret_cast<const char*>(start)), mSize(size)
    {}

    const unsigned char * const &data() const { return m_data; }
    const char * const &signedData() const { return m_signedData; }
    const size_t &size() const { return mSize; }

    const unsigned char *begin() const { return m_data; }
    const unsigned char *end() const { return m_data + mSize; }

private:
    const unsigned char *m_data;
    const char *m_signedData;
    size_t mSize;
};

/**
 * @brief uses inline assembler to add a resource to the translation unit. Needs to be called from global scope.
 */
#define ADD_RESOURCE(NAME,PATH) INCBIN(NAME, PATH)

/**
 * @brief add resources that is already addded in another compilation unit
 */
#define ADD_EXT_RESOURCE(NAME) INCBIN_EXTERN(NAME)

/**
 * @brief loads a previously added resource in the current scope (returns a resource object)
 */
#define LOAD_RESOURCE(NAME) ([]() \
{               \
    return mpu::Resource((g##NAME##Data), g##NAME##Size);          \
})()


}

#endif //MPUTILS_RESOURCE_H
