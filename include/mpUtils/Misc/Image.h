/*
 * mpUtils
 * imageLoading.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2019 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_IMAGE_H
#define MPUTILS_IMAGE_H

// includes
//--------------------
#include <memory>
#include <string>
#include <vector>
#include <mpUtils/external/stb_image_write.h>
#include "mpUtils/external/stb_image.h"
#include "mpUtils/Log/Log.h"
//--------------------

// namespace
//--------------------
namespace mpu {
//--------------------

//-------------------------------------------------------------------
/**
 * @brief class representing an image as an array of pixels on the cpu
 *      Indexing starts in the lower left of the image (by opengl convention).
 * @tparam storeT the type used by every element of a pixel
 */
template <typename storeT>
class Image
{
public:
    using InternalType = storeT;

    // construction / loading
    Image(); //!< create image without data
    Image(int width, int height, int channels); //!< create empty image of given size
    template <typename CopyDataType>
    Image(const CopyDataType* pixels, int width, int height, int channels); //!< create image by copying pixel data, pixels is expected of size width*height*channels
    template <typename ItT>
    Image(ItT pixelsBegin, ItT pixelsEnd, int width, int height, int channels); //!< create image by copying pixel data using a pair of iterators, pixels is expected of size width*height*channels
    explicit Image(const std::string& filename, int forceChannels = STBI_rgb_alpha); //!< load image from file using stb image. When loading non float data into an image of float type data is automatically normalized to 0,1 range
    Image(const unsigned char * data, int length, int forceChannels = STBI_rgb_alpha); //!< load image from file stored in memory.  When loading non float data into an image of float type data is automatically normalized to 0,1 range

    // conversion
    template <typename otherT, std::enable_if< true, int>::type = 0>
    explicit operator Image<otherT>(); //!< convert to other images, correctly normalizes between 8bit 18bit and 32 bit images

    // storing
    void storePNG(const std::string& filename); //!< stores the image in a png file
    void storeTGA(const std::string& filename); //!< stores the image in a tga file
    void storeJPG(const std::string& filename, int quality); //!< stores image as a jpeg file

    // size checking
    int length() const {return m_length;} //!< length of the image in memory (width*height*channels)
    int width() const {return m_width;} //!< width of the image in pixel
    int height() const {return m_height;} //!< height of the image in pixel
    int channels() const {return m_channels;} //!< number of channels in the image

    // data access
    struct rowProxy //!< row struct allows acces to collums of a row
    {
    public:
        rowProxy(InternalType* data, int channels) : rowData(data), channels(channels) {}
        InternalType *operator[](int col) { return &rowData[col * channels]; } //!< access a column
        const InternalType *operator[](int col) const { return &rowData[col * channels]; } //!< access a column
    private:
        InternalType* rowData;
        const int channels;
    };
    struct constRowProxy //!< row struct allows acces to collums of a const row
    {
    public:
        constRowProxy(const InternalType* data, int channels) : rowData(data), channels(channels) {}
        const InternalType *operator[](int col) const { return &rowData[col * channels]; } //!< access a column
    private:
        const InternalType* rowData;
        const int channels;
    };

    rowProxy operator[](int row) { return rowProxy( &m_data[row*m_width*m_channels],m_channels); } //!< access a row
    constRowProxy operator[](int row) const { return constRowProxy(&m_data[row*m_width*m_channels],m_channels); } //!< access a row

    InternalType &operator()(int idx) { return m_data[idx]; } //!< access value
    const InternalType &operator()(int idx) const { return m_data[idx]; } //!< access value

    InternalType* T(int row, size_t col) { return &this[col][row];} //!< access pixel as if matrix was transposed
    const InternalType* T(int row, size_t col) const { return &this[col][row];} //!< access pixel as if matrix was transposed

    InternalType* data() {return m_data.data();} //!< direct access to the internal data
    const InternalType* data() const {return m_data.data();} //!< direct access to the internal data

    // operations on the pixels
    void setZero(int row, int col, int maxRow, int maxCol, int channel=-1); //!< sets the entire image data to zero, or a specific channel if given
    Image cloneSubregion(int row, int col, int maxRow, int maxCol) const; //!< returns a new image containing a subregion of the current image
    void normalize(InternalType maxValue); //!< divide every value in the image by max value

private:
    std::vector<InternalType> m_data;
    int m_width;
    int m_height;
    int m_channels;
    int m_length;
};

// some default image specializations
using Image8 = Image<uint8_t>;
using Image16 = Image<uint16_t>;
using Image32 = Image<float>;

// template function definition
//-------------------------------------------------------------------

template <typename storeT>
Image<storeT>::Image()
        : m_data(), m_width(0), m_height(0), m_channels(0), m_length(0)
{
}

template <typename storeT>
Image<storeT>::Image(int width, int height, int channels)
        : m_width(width), m_height(height), m_channels(channels), m_length(width*height*channels), m_data(width*height*channels)
{
}

template <typename storeT>
template <typename CopyDataType>
Image<storeT>::Image(const CopyDataType* pixels, int width, int height, int channels)
        : m_width(width), m_height(height), m_channels(channels), m_length(width*height*channels), m_data(pixels,pixels+m_length)
{
}

template <typename storeT>
template <typename ItT>
Image<storeT>::Image(const ItT pixelsBegin, const ItT pixelsEnd, int width, int height, int channels)
        : m_width(width), m_height(height), m_channels(channels), m_length(width*height*channels), m_data(pixelsBegin,pixelsEnd)
{
}

template <typename storeT>
Image<storeT>::Image(const std::string& filename, int forceChannels)
        : Image()
{
    stbi_set_flip_vertically_on_load(true);

    if(stbi_is_hdr(filename.c_str()))
    {
        if(sizeof(InternalType) < 4 || !std::is_floating_point<InternalType>())
            logWARNING("Image") << "Image " << filename << " is 32 bit float, but internal type of image is smaller or not float!";

        int w;
        int h;
        int c;
        auto result = std::unique_ptr<float[], decltype(&stbi_image_free)>(stbi_loadf(filename.c_str(), &w, &h, &c, forceChannels), &stbi_image_free);
        assert_critical(result,"Image","Error while loading image file " + filename + " reason: " + stbi_failure_reason());
        m_width = w;
        m_height = h;
        m_channels = forceChannels == 0 ? c : forceChannels;
        m_length = m_width*m_height*m_channels;
        m_data.assign(result.get(),result.get()+m_length);
    }
    else if(stbi_is_16_bit(filename.c_str()))
    {
        if(sizeof(InternalType) < 2)
            logWARNING("Image") << "Image " << filename << " is 16 bit, but internal type of image is smaller!";

        int w;
        int h;
        int c;
        auto result = std::unique_ptr<unsigned short[], decltype(&stbi_image_free)>(stbi_load_16(filename.c_str(), &w, &h, &c, forceChannels), &stbi_image_free);
        assert_critical(result,"Image","Error while loading image file " + filename + " reason: " + stbi_failure_reason());
        m_width = w;
        m_height = h;
        m_channels = forceChannels == 0 ? c : forceChannels;
        m_length = m_width*m_height*m_channels;
        m_data.assign(result.get(),result.get()+m_length);

        if(std::is_floating_point<InternalType>())
            normalize(65535);
    }
    else
    {
        int w;
        int h;
        int c;
        auto result = std::unique_ptr<unsigned char[], decltype(&stbi_image_free)>(stbi_load(filename.c_str(), &w, &h, &c, forceChannels), &stbi_image_free);
        assert_critical(result,"Image","Error while loading image file " + filename + " reason: " + stbi_failure_reason());
        m_width = w;
        m_height = h;
        m_channels = forceChannels == 0 ? c : forceChannels;
        m_length = m_width*m_height*m_channels;
        m_data.assign(result.get(),result.get()+m_length);

        if(std::is_floating_point<InternalType>())
            normalize(255);
    }
}

template <typename storeT>
Image<storeT>::Image(const unsigned char* data, int length, int forceChannels)
{
    stbi_set_flip_vertically_on_load(true);

    if(stbi_is_hdr_from_memory(data,length))
    {
        if(sizeof(InternalType) < 4 || !std::is_floating_point<InternalType>())
            logWARNING("Image") << "Image loaded from data is 32 bit float, but internal type of image is smaller or not float!";

        int w;
        int h;
        int c;
        auto result = std::unique_ptr<float[], decltype(&stbi_image_free)>(stbi_loadf_from_memory(data,length, &w, &h, &c, forceChannels), &stbi_image_free);
        assert_critical(result,"Image", std::string("Error while loading image from data reason: ") + stbi_failure_reason());
        m_width = w;
        m_height = h;
        m_channels = forceChannels == 0 ? c : forceChannels;
        m_length = m_width*m_height*m_channels;
        m_data.assign(result.get(),result.get()+m_length);
    }
    else if(stbi_is_16_bit_from_memory(data,length))
    {
        if(sizeof(InternalType) < 2)
            logWARNING("Image") << "Image loaded from data is 16 bit, but internal type of image is smaller!";

        int w;
        int h;
        int c;
        auto result = std::unique_ptr<unsigned short[], decltype(&stbi_image_free)>(stbi_load_16_from_memory(data,length, &w, &h, &c, forceChannels), &stbi_image_free);
        assert_critical(result,"Image", std::string("Error while loading image from data reason: ") + stbi_failure_reason());
        m_width = w;
        m_height = h;
        m_channels = forceChannels == 0 ? c : forceChannels;
        m_length = m_width*m_height*m_channels;
        m_data.assign(result.get(),result.get()+m_length);

        if(std::is_floating_point<InternalType>())
            normalize(65535);
    }
    else
    {
        int w;
        int h;
        int c;
        auto result = std::unique_ptr<unsigned char[], decltype(&stbi_image_free)>(stbi_load_from_memory(data,length, &w, &h, &c, forceChannels), &stbi_image_free);
        assert_critical(result,"Image", std::string("Error while loading image from data reason: ") + stbi_failure_reason());
        m_width = w;
        m_height = h;
        m_channels = forceChannels == 0 ? c : forceChannels;
        m_length = m_width*m_height*m_channels;
        m_data.assign(result.get(),result.get()+m_length);

        if(std::is_floating_point<InternalType>())
            normalize(255);
    }
}

template <typename storeT>
void Image<storeT>::normalize(InternalType maxValue)
{
    for(auto& pixelChannel : m_data)
        pixelChannel = pixelChannel/maxValue;
}

template <typename storeT>
void Image<storeT>::setZero(int row, int col, int maxRow, int maxCol, int channel)
{
    if(channel < 0)
        for(int i=row;i<maxRow;++i)
            for(int j=col;j<maxCol;++j)
                for(int c=0;c<m_channels;++c)
                    m_data[(i*m_width+j)*m_channels+c] = 0;
    else
        for(int i=row;i<maxRow;++i)
            for(int j=col;j<maxCol;++j)
                m_data[(i*m_width+j)*m_channels+channel] = 0;
}

template <typename storeT>
Image<storeT> Image<storeT>::cloneSubregion(int row, int col, int maxRow, int maxCol) const
{
    Image<storeT> result(maxCol-col,maxRow-row,m_channels);
    for(int i=row;i<maxRow;++i)
        for(int j=col;j<maxCol;++j)
            for(int c=0;c<m_channels;++c)
                result[i-row][j-col][c] = (*this)[i][j][c];
    return result;
}

template <typename storeT>
void Image<storeT>::storePNG(const std::string& filename)
{
    stbi_flip_vertically_on_write(true);
    int r;
    if(std::is_floating_point<storeT>() || sizeof(InternalType) >1)
    {
        auto temp = Image<unsigned char>(*this);
        r = stbi_write_png(filename.c_str(),m_width,m_height,m_channels,temp.data(),sizeof(Image<unsigned char>::InternalType)*m_channels*m_width);
    } else
    {
        r = stbi_write_png(filename.c_str(),m_width,m_height,m_channels,m_data.data(),sizeof(InternalType)*m_channels*m_width);
    }
    assert_critical(r!=0,"Image","Failed to store image file: " + filename);
}

template <typename storeT>
void Image<storeT>::storeTGA(const std::string& filename)
{
    stbi_flip_vertically_on_write(true);
    int r;
    if(std::is_floating_point<storeT>() || sizeof(InternalType) >1)
    {
        auto temp = Image<unsigned char>(*this);
        r = stbi_write_tga(filename.c_str(),m_width,m_height,m_channels,temp.data());
    } else
    {
        r = stbi_write_tga(filename.c_str(),m_width,m_height,m_channels,m_data.data());
    }
    assert_critical(r!=0,"Image","Failed to store image file: " + filename);
}

template <typename storeT>
void Image<storeT>::storeJPG(const std::string& filename, int quality)
{
    stbi_flip_vertically_on_write(true);
    int r;
    if(std::is_floating_point<storeT>() || sizeof(InternalType) >1)
    {
        auto temp = Image<unsigned char>(*this);
        r = stbi_write_jpg(filename.c_str(),m_width,m_height,m_channels,temp.data(),quality);
    } else
    {
        r = stbi_write_jpg(filename.c_str(),m_width,m_height,m_channels,m_data.data(),quality);
    }
    assert_critical(r!=0,"Image","Failed to store image file: " + filename);
}

template <typename storeT>
template <typename otherT, std::enable_if<true, int>::type>
Image<storeT>::operator Image<otherT>()
{
    Image<otherT> result(m_width,m_height,m_channels);

    float multi = 1.0;
    if(std::is_floating_point<storeT>() && !std::is_floating_point<otherT>() && sizeof(otherT) <= 1)
        multi = 255.0;
    else if(std::is_floating_point<storeT>() && !std::is_floating_point<otherT>() && sizeof(otherT) >= 2)
        multi = 65535.0;
    else if(std::is_floating_point<otherT>() && !std::is_floating_point<storeT>() && sizeof(storeT) <= 1)
        multi = 1.0/255.0;
    else if(std::is_floating_point<otherT>() && !std::is_floating_point<storeT>() && sizeof(storeT) >= 2)
        multi = 1.0/65535.0;
    else if( sizeof(otherT) <= 1 && sizeof(storeT) >= 2)
        multi = 255.0/65535.0;
    else if(sizeof(storeT) <= 1 && sizeof(otherT) >= 2)
        multi = 65535.0/255.0;

    for(int i=0;i<m_height;++i)
        for(int j=0;j<m_width;++j)
            for(int c=0;c<m_channels;++c)
                result[i][j][c] = std::min(multi* float((*this)[i][j][c]),float(std::numeric_limits<otherT>::max()));

    return result;
}

}
#endif //MPUTILS_IMAGE_H
