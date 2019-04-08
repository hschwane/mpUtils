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
 * Use the ADD_RESOURCES cmake function in cmake/addResource.cmake to embed a resource file into a executable.
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

    const char * const &data() const { return mData; }
    const size_t &size() const { return mSize; }

    const char *begin() const { return mData; }
    const char *end() const { return mData + mSize; }

private:
    const char *mData;
    size_t mSize;
};


#define LOAD_RESOURCE(x) ([]() \
{                                            \
    extern const char _binary_##x##_start, _binary_##x##_end;           \
    return mpu::Resource(&_binary_##x##_start, &_binary_##x##_end);          \
})()


}

#endif //MPUTILS_RESOURCE_H
