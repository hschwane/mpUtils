/*
 * gpulic
 * Texture.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Texture class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#ifndef GPULIC_TEXTURE_H
#define GPULIC_TEXTURE_H

// includes
//--------------------
#include <GL/glew.h>
#include <glm/glm.hpp>
#include "mpUtils/Graphics/Opengl/Sampler.h"
#include <cinttypes>
#include <string>
#include <memory>
#include <vector>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * @brief supported types of textures
 */
enum class TextureTypes : GLenum
{
    texture1D = GL_TEXTURE_1D,
    texture2D = GL_TEXTURE_2D,
    texture3D = GL_TEXTURE_3D,
    texture1DArray = GL_TEXTURE_1D_ARRAY,
    texture2DArray = GL_TEXTURE_2D_ARRAY,
    cubemap = GL_TEXTURE_CUBE_MAP,
    cubemapArray = GL_TEXTURE_CUBE_MAP_ARRAY,
    multisample2D = GL_TEXTURE_2D_MULTISAMPLE,
    multisample2DArray = GL_TEXTURE_2D_MULTISAMPLE_ARRAY,
};

//-------------------------------------------------------------------
/**
 * @brief identify faces of a cubemap
 */
namespace CubeMapFace {
enum cmf
{
    posX = 0,
    negX,
    posY,
    negY,
    posZ,
    negZ
};
}

//-------------------------------------------------------------------
/**
 * class Texture
 *
 * Wraps a openGL Texture with immutable storage. Since it is an immutable object creating a texture without allocating
 * memory does not make sense and there is no default constructor. (this also eliminates the need for runtime checks)
 * You can however use pointers to "Texture" objects.
 * There are many different types of textures and even more different ways to create them,
 * so a number of factory functions is provided below.
 *
 * usage:
 * You can create a texture and allocate memory in its constructor. Then you upload image data using one of the upload functions.
 * It is however recommended to use one of the factory functions below
 * You can then create mipmaps if you need to and bind the texture for use.
 * You can also bind the texture as an image for write access from the shader.
 * To control sampling parameters use the "Sampler" class.
 *
 * Available factory functions include:
 * makeTexture1D / 2D / 3D
 * makeTextureFromFile, makeTextureFromFileHDR, makeTextureFromData, makeTextureFromDataHDR
 * makeTexture1DArray, makeTexture2DArray, makeTexture2DArrayFromFile, makeTexture2DArrayFromFileHDR,
 * makeCubemap, makeCubemapFromFile, makeCubemapFromFileHDR
 *
 */
class Texture
{
public:
    /**
     * @brief Construct a texture and allocates memory for it. Values not supported by the chosen texture type will be ignored.
     * @param type The type of texture, see TextureTypes.
     * @param internalFormat the internal pixel format (see openGL documentation)
     * @param levels the number of mipMap levels
     * @param width with of the texture in pixel
     * @param height high of the texture in pixel
     * @param depth number of layers for 3d textures
     * @param samples number of samples for multisampling textures
     */
    Texture(TextureTypes type, GLenum internalFormat, int levels, int width, int height = 1, int depth = 1, int samples=0);

    ~Texture(); //!< destructor
    operator uint32_t() const; //!< convert to opengl handle for use in raw opengl functions

    // make Texture non copyable but movable
    Texture(const Texture& other) = delete;
    Texture& operator=(const Texture& other) = delete;
    Texture(Texture&& other) noexcept : m_textureHandle(0){*this = std::move(other);};
    Texture& operator=(Texture&& other) noexcept;

    /**
     * @brief Upload texture data for an entire mipMap level of the texture
     * @param data Pointer to the data that should be uploaded.
     * @param format input format of the pixel data:
     *          GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA,
     *          GL_BGRA, GL_DEPTH_COMPONENT, and GL_STENCIL_INDEX
     * @param type data type of the data that is uploaded defaults to GL_UNSIGNED_BYTE
     * @param level the mipmap level to upload to, defaults to 0
     */
    void uploadData( const GLvoid* data, GLenum format, GLenum type = GL_UNSIGNED_BYTE, int level = 0);

    /**
     * @brief Upload one entire 1D row to a 2D texture / 1D Array texture
     * @param data Pointer to the data that should be uploaded.
     * @param layer the layer in which to upload the data
     * @param format input format of the pixel data:
     *          GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA,
     *          GL_BGRA, GL_DEPTH_COMPONENT, and GL_STENCIL_INDEX
     * @param type data type of the data that is uploaded defaults to GL_UNSIGNED_BYTE
     * @param level the mipmap level to upload to, defaults to 0
     */
    void uploadRow( const GLvoid* data, int row, GLenum format, GLenum type = GL_UNSIGNED_BYTE, int level = 0);

    /**
     * @brief Upload one entire 2D layer to a 3D texture / 2D array texture, or a cubemap face to a cubemap or cubemap array.
     *          When uploading cubemap faces you can use the enum in the "CubeMapFace" namespace to spcify what face to upload.
     * @param data Pointer to the data that should be uploaded.
     * @param layer the layer in which to upload the data or cubemap face
     * @param format input format of the pixel data:
     *          GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA,
     *          GL_BGRA, GL_DEPTH_COMPONENT, and GL_STENCIL_INDEX
     * @param type data type of the data that is uploaded defaults to GL_UNSIGNED_BYTE
     * @param level the mipmap level to upload to, defaults to 0
     */
    void upload2DLayer( const GLvoid* data, int layer, GLenum format, GLenum type = GL_UNSIGNED_BYTE, int level = 0);

    /**
     * @brief Partially upload data, specify the size of the data as well as a offset from the origin of the texture.
     * @param data the data to upload
     * @param format input format of the pixel data:
     *          GL_COLOR_INDEX, GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA,
     *          GL_RGB, GL_BGR, GL_RGBA, GL_BGRA,
     *          GL_LUMINANCE, and GL_LUMINANCE_ALPHA
     * @param type  data type of the data that is uploaded defaults eg GL_UNSIGNED_BYTE
     * @param width with of the changed data
     * @param height height of the changed data
     * @param depth depth of the changed data
     * @param xoffset offset from the texture origin in x direction
     * @param yoffset offset from the texture origin in y direction
     * @param zoffset offset from the origin in z direction
     * @param level the mipmap level to change
     */
    void uploadSubData(const GLvoid* data, GLenum format, GLenum type, int width, int height=1, int depth=1, int xoffset =0, int yoffset =0, int zoffset =0, int level = 0);

    /**
     * @brief generate all needed mipmaps
     */
    void generateMipmaps();

    /**
     * @brief clear the texture data
     * @param clearData data for one pixel to be used as a clear value
     * @param format the pixel format: GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_DEPTH_COMPONENT, or GL_STENCIL_INDEX
     * @param type the type of the pixel data, eg  GL_UNSIGNED_BYTE, GL_INT or GL_FLOAT
     * @param level the mipmap level to clear
     */
    void clear(const void* clearData, GLenum format, GLenum type = GL_FLOAT, GLint level=0);

    /**
     * @brief binds the txture to a texture unit, after activating that unit
     * @param textureUnit
     */
    void bind(GLenum textureUnit) const;

    /**
     * @brief Bind Texture to an image object for wirte access in the shader
     * @param bindingIndex The Image binding index where to bind the texture
     * @param access GL_READ_ONLY, GL_WRITE_ONLY, or GL_READ_WRITE
     * @param format specifies the format used for formatted stores from the shader
     * @param level mipmap level of the texture to be bound
     * @param layered bind multiple layers of a 3d texture / texture array?
     * @param layer the layer to bind if layered is false
     */
    void bindImage(GLuint bindingIndex, GLenum access, GLenum format, GLint level=0, bool layered=false, GLint layer=0) const;

    /**
     * @brief Generates a texture handle to access this texture in bindless mode using the sampling settings from "sampler"
     *          Do not forget to make your handle resident using glMakeTextureHandleResidentARB !
     * @param sampler the sampler to use the settings from
     * @return the texture handle to use in the shader
     */
    GLuint64 getTextureHandle(const Sampler& sampler);

    /**
     * @brief Generates a handle to access this texture for storing data in bindless mode#
     *          Do not forget to make your handle resident using glMakeImageHandleResidentARB !
     * @param format specifies the format used for formatted stores from the shader
     * @param level mipmap level of the texture to be bound
     * @param layered bind multiple layers of a 3d texture / texture array?
     * @param layer the layer to bind if layered is false
     * @return the image handle to use in the shader
     */
    GLuint64 getImageHandle( GLenum format, GLint level=0, bool layered=false, GLint layer=0);

    int32_t width() const { return m_width;} //!< get the width of the texture
    int32_t height() const { return m_height;} //!< get the height of the texture
    int32_t depth() const { return m_depth;} //!< get the depth of the texture (if it is 3D)
    int32_t levels() const { return m_levels;} //!< get the amount of mipmap levels
    TextureTypes type() const { return m_type;} //!< get the type of texture we are dealing with
    GLenum internalFormat() const { return m_internalFormat;} //!< get the internal format

    static uint32_t maxMipmaps(uint32_t width, uint32_t height, uint32_t depth); //!< calculates the maximum number of mipmaps for given image dimensions

private:
    uint32_t m_textureHandle;

    GLint m_width; //!< the with of the texture
    GLint m_height; //!< the height of the texture
    GLint m_depth; //!< the number of layers in 3D textures
    GLint m_levels; //!< the amount of mipmap levels
    TextureTypes m_type; //!< the target this texture will be bound to when binding
    GLenum m_internalFormat; //!< the internal format of the texture
};


//-------------------------------------------------------------------
// factory functions for texture creation

/**
 * @brief Generates a 1D texture optionally data can be uploaded to level 0. If you want to upload to multiple levels you have to do it later by yourself
 * @param size the size of the texture (number of pixels)
 * @param internalFormat the internal format of the texture
 * @param data raw data to upload into level 0 of the texture (pass nullptr to not upload anything)
 * @param format format of the data that will be uploaded (if any)
 * @param type data type of the uploaded data
 * @param levels number of mipmap levels. 0 means auto detect number of mipmap levels.
 * @param genMipmaps automatically generate mipmaps after uploading
 * @return returns a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTexture1D(int size, GLenum internalFormat, const GLvoid* data = nullptr, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, int levels=0, bool genMipmaps = true);

/**
 * @brief Generates a 2D texture optionally data can be uploaded to level 0. If you want to upload to multiple levels you have to do it later by yourself
 * @param width the width of the texture (number of pixels)
 * @param height the height of the texture
 * @param internalFormat the internal format of the texture
 * @param data raw data to upload into level 0 of the texture (pass nullptr to not upload anything)
 * @param format format of the data that will be uploaded (if any)
 * @param type data type of the uploaded data
 * @param levels number of mipmap levels. 0 means auto detect number of mipmap levels.
 * @param genMipmaps automatically generate mipmaps after uploading
 * @return returns a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTexture2D(int width, int height, GLenum internalFormat, const GLvoid* data = nullptr, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, int levels=0, bool genMipmaps = true);

/**
 * @brief Generates a 3D texture optionally data can be uploaded to level 0. If you want to upload to multiple levels you have to do it later by yourself
 * @param width the width of the texture (number of pixels)
 * @param height the height of the texture
 * @param depth of the texture
 * @param internalFormat the internal format of the texture
 * @param data raw data to upload into level 0 of the texture (pass nullptr to not upload anything)
 * @param format format of the data that will be uploaded (if any)
 * @param type data type of the uploaded data
 * @param levels number of mipmap levels. 0 means auto detect number of mipmap levels.
 * @param genMipmaps automatically generate mipmaps after uploading
 * @return a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTexture3D(int width, int height, int depth, GLenum internalFormat, const GLvoid* data = nullptr, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, int levels=0, bool genMipmaps = true);

/**
 * @brief makes a 2D texture from an image file
 * @param filename path to the image file
 * @param number of components to fill 3=RGB, 4=RGBA, 1=Gray, 2=Gray+Alpha
 * @param generateMipmaps should mipmaps be generated
 * @return a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTextureFromFile(const std::string& filename, int numComponents = 4, bool generateMipmaps = true);

/**
 * @brief makes a 2D HDR texture from an image file
 * @param filename path to the image file
 * @param number of components to fill 3=RGB, 4=RGBA, 1=Gray, 2=Gray+Alpha
 * @param generateMipmaps should mipmaps be generated
 * @return a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTextureFromFileHDR(const std::string& filename, int numComponents = 4,  bool generateMipmaps = true);

/**
 * @brief makes a 2D texture from an image file already in memory (data is first passed to stb_image for decoding / decompression)
 * @param filename path to the image file
 * @param number of components to fill 3=RGB, 4=RGBA, 1=Gray, 2=Gray+Alpha
 * @param generateMipmaps should mipmaps be generated
 * @return a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTextureFromData( const unsigned char * data, int length, int numComponents = 4, bool generateMipmaps = true);

/**
 * @brief makes a 2D HDR texture from an image file already in memory (data is first passed to stb_image for decoding / decompression)
 * @param filename path to the image file
 * @param number of components to fill 3=RGB, 4=RGBA, 1=Gray, 2=Gray+Alpha
 * @param generateMipmaps should mipmaps be generated
 * @return a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTextureFromDataHDR( const unsigned char * data, int length, int numComponents = 4, bool generateMipmaps = true);

/**
 * @brief Generates an array of 1D textures optionally data can be uploaded to level 0. Data is expected to be big enough to fill the entire array.
 * @param size the size of the texture (number of pixels)
 * @param numberOfLayers number of 1D textures to be stored in the array
 * @param internalFormat the internal format of the texture
 * @param data raw data to upload into level 0 of the texture (pass nullptr to not upload anything)
 * @param format format of the data that will be uploaded (if any)
 * @param type data type of the uploaded data
 * @param levels number of mipmap levels. 0 means auto detect number of mipmap levels.
 * @param genMipmaps automatically generate mipmaps after uploading
 * @return returns a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTexture1DArray(int size, int numberOfLayers, GLenum internalFormat, const GLvoid* data = nullptr, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, int levels=0, bool genMipmaps = true);

/**
 * @brief Generates an array of 2D textures optionally data can be uploaded to level 0. Data is expected to be big enough to fill the entire array.
 * @param width the width of the texture (number of pixels)
 * @param height the height of the texture
 * @param numberOfLayers number of 1D textures to be stored in the array
 * @param internalFormat the internal format of the texture
 * @param data raw data to upload into level 0 of the texture (pass nullptr to not upload anything)
 * @param format format of the data that will be uploaded (if any)
 * @param type data type of the uploaded data
 * @param levels number of mipmap levels. 0 means auto detect number of mipmap levels.
 * @param genMipmaps automatically generate mipmaps after uploading
 * @return returns a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTexture2DArray(int width, int height, int numberOfLayers, GLenum internalFormat, const GLvoid* data = nullptr, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, int levels=0, bool genMipmaps = true);

/**
 * @brief makes an array of 2D textures from a list of image files
 * @param files list of paths to the imag files
 * @param number of components to fill 3=RGB, 4=RGBA, 1=Gray, 2=Gray+Alpha
 * @param generateMipmaps should mipmaps be generated
 * @return a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTexture2DArrayFromFiles(const std::vector<std::string>& files, int numComponents = 4, bool generateMipmaps = true);

/**
 * @brief makes an array of 2D HDR textures from a list of image files
 * @param files list of paths to the imag files
 * @param number of components to fill 3=RGB, 4=RGBA, 1=Gray, 2=Gray+Alpha
 * @param generateMipmaps should mipmaps be generated
 * @return a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeTexture2DArrayFromFilesHDR(const std::vector<std::string>& files, int numComponents = 4, bool generateMipmaps = true);

/**
 * @brief Generates a cube map texture. Optionally data can be uploaded to level 0. Data is expected to be big enough to fill all 6 faces.
 * @param size with / height of a single cube map face (cube map faces need to be squares)
 * @param internalFormat the internal format of the texture
 * @param data raw data to upload into level 0 of the texture (pass nullptr to not upload anything)
 * @param format format of the data that will be uploaded (if any)
 * @param type data type of the uploaded data
 * @param levels number of mipmap levels. 0 means auto detect number of mipmap levels.
 * @param genMipmaps automatically generate mipmaps after uploading
 * @return returns a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeCubemap(int size, GLenum internalFormat, const GLvoid* data = nullptr, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, int levels=0, bool genMipmaps = true);

/**
 * @brief makes a cube map from a list of 6 files
 * @param files list of paths to the imag files
 * @param number of components to fill 3=RGB, 4=RGBA, 1=Gray, 2=Gray+Alpha
 * @param generateMipmaps should mipmaps be generated
 * @return a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeCubemapFromFiles(const std::vector<std::string>& files, int numComponents = 4, bool generateMipmaps = true);

/**
 * @brief makes a HDR cube map from a list of 6 files
 * @param files list of paths to the imag files
 * @param number of components to fill 3=RGB, 4=RGBA, 1=Gray, 2=Gray+Alpha
 * @param generateMipmaps should mipmaps be generated
 * @return a unique pointer to the resulting texture object
 */
std::unique_ptr<Texture> makeCubemapFromFilesHDR(const std::vector<std::string>& files, int numComponents = 4, bool generateMipmaps = true);

}}

#endif //GPULIC_TEXTURE_H
