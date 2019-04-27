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
 * Use the ADD_RESOURCES makro to embed a resource file into a executable.
 * You can set the search path by using the compiler option -Wa,-I/some/path in your cmake
 * Then use the macro below to load the resource. It will return an object of the following class.
 * You can then use the iterators as well as data and size member functions to access the data.
 *
 * example:
 * int main(void)
 * {
 *   Resource text = LOAD_RESOURCE(resources_test_txt);
 *   std::cout << std::string(text.data(), text.size()) << std::endl;
 * }
 * where the file was located at resources/test.txt relative to the cmake file from where ADD_RESOURCES was called
 *
 */
class Resource
{
public:
    Resource(const char *start, const char *end): mData(start),
                                                  mSize(end - start)
    {}

    Resource(const char *start, size_t size): mData(start), mSize(size)
    {}

    const char * const &data() const { return mData; }
    const size_t &size() const { return mSize; }

    const char *begin() const { return mData; }
    const char *end() const { return mData + mSize; }

private:
    const char *mData;
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
    return mpu::Resource(reinterpret_cast<const char *>(g##NAME##Data), g##NAME##Size);          \
})()


}

#endif //MPUTILS_RESOURCE_H
