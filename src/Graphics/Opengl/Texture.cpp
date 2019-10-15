/*
 * gpulic
 * Texture.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Texture class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Graphics/Opengl/Texture.h"
#include "mpUtils/Misc/imageLoading.h"
#include "mpUtils/Log/Log.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Texture class
//-------------------------------------------------------------------


Texture::Texture(TextureTypes type, GLenum internalFormat, int levels, int width, int height, int depth, int samples)
        : m_internalFormat(internalFormat), m_width(width), m_height(height), m_depth(depth), m_levels(levels),
          m_type(type)
{
    glCreateTextures(static_cast<GLenum>(type), 1, &m_textureHandle);

    switch(type)
    {
        case TextureTypes::texture1D:
            glTextureStorage1D(*this, levels, internalFormat, width);
            break;

        case TextureTypes::cubemap:
            glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
            assert_true(width == height, "Texture", "Cubemap textures must be squares!");
        case TextureTypes::texture1DArray:
        case TextureTypes::texture2D:
            glTextureStorage2D(m_textureHandle, levels, internalFormat, width, height);
            break;

        case TextureTypes::cubemapArray:
            glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
            assert_true(width == height, "Texture", "Cubemap textures must be squares!");
        case TextureTypes::texture2DArray:
        case TextureTypes::texture3D:
            glTextureStorage3D(*this, levels, internalFormat, width, height, depth);
            break;

        case TextureTypes::multisample2D:
            glTextureStorage2DMultisample(*this, samples, samples, width, height, GL_TRUE);
            break;

        case TextureTypes::multisample2DArray:
            glTextureStorage3DMultisample(*this, samples, samples, width, height, depth, GL_TRUE);
            break;

        default:
            assert_true(false, "Texture", "Unknown texture type!");
    }
}

Texture::~Texture()
{
    if(m_textureHandle != 0)
        glDeleteTextures(1, &m_textureHandle);
}

Texture& Texture::operator=(Texture&& other) noexcept
{
    using std::swap;
    swap(m_textureHandle,other.m_textureHandle);
    swap(m_width,other.m_width);
    swap(m_height,other.m_height);
    swap(m_depth,other.m_depth);
    swap(m_levels,other.m_levels);
    swap(m_type,other.m_type);
    swap(m_internalFormat,other.m_internalFormat);
    return *this;
}

Texture::operator uint32_t() const
{
    return m_textureHandle;
}

void Texture::uploadData(const GLvoid* data, GLenum format, GLenum type, int level)
{
    uploadSubData(data, format, type, m_width, m_height, m_depth, 0, 0, 0, level);
}

void Texture::uploadRow(const GLvoid* data, int row, GLenum format, GLenum type, int level)
{
    uploadSubData(data, format, type, m_width, 1, 1, 0, row, 0, level);
}

void Texture::upload2DLayer(const GLvoid* data, int layer, GLenum format, GLenum type, int level)
{
    uploadSubData(data, format, type, m_width, m_height, 1, 0, 0, layer, level);
}

void
Texture::uploadSubData(const GLvoid* data, GLenum format, GLenum type, int width, int height, int depth, int xoffset,
                       int yoffset, int zoffset, int level)
{
    switch(m_type)
    {
        case TextureTypes::texture1D:
            glTextureSubImage1D(*this, level, xoffset, width, format, type, data);
            break;


        case TextureTypes::texture1DArray:
        case TextureTypes::texture2D:
            glTextureSubImage2D(*this, level, xoffset, yoffset, width, height, format, type, data);
            break;

        case TextureTypes::cubemap:
        case TextureTypes::cubemapArray:
        case TextureTypes::texture2DArray:
        case TextureTypes::texture3D:
            glTextureSubImage3D(*this, level, xoffset, yoffset, yoffset, width, height, depth, format, type, data);
            break;

        case TextureTypes::multisample2D:
            logWARNING("Texture") << "Trying to upload data to a multisampling texture!";
            assert_true(false, "Texture", "Trying to upload data to a multisampling texture!");
            break;

        case TextureTypes::multisample2DArray:
            logWARNING("Texture") << "Trying to upload data to a multisampling texture!";
            assert_true(false, "Texture", "Trying to upload data to a multisampling texture!");
            break;

        default:
            assert_true(false, "Texture", "Unknown texture type!");
    }
}

void Texture::generateMipmaps()
{
    glGenerateTextureMipmap(*this);
}

void Texture::bind(GLenum textureUnit) const
{
    glActiveTexture(textureUnit);
    glBindTexture(static_cast<GLenum>(m_type), *this);
}

void Texture::bindImage(GLuint bindingIndex, GLenum access, GLenum format, GLint level, bool layered, GLint layer) const
{
    glBindImageTexture(bindingIndex, *this, level, layered, layer, access, format);
}

void Texture::clear(const void* clearData, GLenum format, GLenum type, GLint level)
{
    glClearTexImage(*this, level, format, type, clearData);
}

inline uint32_t Texture::maxMipmaps(const uint32_t width, const uint32_t height, const uint32_t depth)
{
    return static_cast<uint32_t>(1 + glm::floor(glm::log2(
            glm::max(static_cast<float>(width), glm::max(static_cast<float>(height), static_cast<float>(depth))))));
}

GLuint64 Texture::getTextureHandle(const Sampler& sampler)
{
    return glGetTextureHandleARB(m_textureHandle);
}

GLuint64 Texture::getImageHandle(GLenum format, GLint level, bool layered, GLint layer)
{
    return glGetImageHandleARB(m_textureHandle, level, layered, layer, format);
}

// texture factory functions

std::unique_ptr<Texture>
makeTexture1D(int size, GLenum internalFormat, const void* data, GLenum format, GLenum type, int levels,
              bool genMipmaps)
{
    if(levels == 0)
        levels = Texture::maxMipmaps(size, 1, 1);

    auto tex = std::make_unique<Texture>(TextureTypes::texture1D, internalFormat, levels, size);

    if(data)
        tex->uploadData(data, format, type);

    if(genMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeTexture2D(int width, int height, GLenum internalFormat, const void* data, GLenum format, GLenum type, int levels,
              bool genMipmaps)
{
    if(levels == 0)
        levels = Texture::maxMipmaps(width, height, 1);

    auto tex = std::make_unique<Texture>(TextureTypes::texture2D, internalFormat, levels, width, height);

    if(data)
        tex->uploadData(data, format, type);

    if(genMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeTexture3D(int width, int height, int depth, GLenum internalFormat, const void* data, GLenum format, GLenum type,
              int levels, bool genMipmaps)
{
    if(levels == 0)
        levels = Texture::maxMipmaps(width, height, depth);

    auto tex = std::make_unique<Texture>(TextureTypes::texture3D, internalFormat, levels, width, height, depth);

    if(data)
        tex->uploadData(data, format, type);

    if(genMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture> makeTextureFromFile(const std::string& filename, int numComponents, bool generateMipmaps)
{
    int width;
    int height;
    auto pixels = loadImageFile(filename, width, height, numComponents);

    int levels = 1;
    if(generateMipmaps)
        levels = Texture::maxMipmaps(width, height, 1);

    GLenum internalFormat = 0;
    GLenum inputFormat = 0;
    if(numComponents == 1)
    {
        internalFormat = GL_R8;
        inputFormat = GL_RED;
    }
    else if(numComponents == 2)
    {
        internalFormat = GL_RG8;
        inputFormat = GL_RG;
    }
    else if(numComponents == 3)
    {
        internalFormat = GL_RGB8;
        inputFormat = GL_RGB;
    }
    else if(numComponents == 4)
    {
        internalFormat = GL_RGBA8;
        inputFormat = GL_RGBA;
    }
    else
    {
        logERROR("Texture") << "numComponents needs to be between 1 and 4!";
        logFlush();
        throw std::logic_error("numComponents needs to be between 1 and 4!");
    }

    auto tex = std::make_unique<Texture>(TextureTypes::texture2D, internalFormat, levels, width, height);
    tex->uploadData(pixels.get(), inputFormat, GL_UNSIGNED_BYTE);

    if(generateMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture> makeTextureFromFileHDR(const std::string& filename, int numComponents, bool generateMipmaps)
{
    int width;
    int height;
    auto pixels = loadImageFileHDR(filename, width, height, numComponents);

    int levels = 1;
    if(generateMipmaps)
        levels = Texture::maxMipmaps(width, height, 1);

    GLenum internalFormat = 0;
    GLenum inputFormat = 0;
    if(numComponents == 1)
    {
        internalFormat = GL_R32F;
        inputFormat = GL_RED;
    }
    else if(numComponents == 2)
    {
        internalFormat = GL_RG32F;
        inputFormat = GL_RG;
    }
    else if(numComponents == 3)
    {
        internalFormat = GL_RGB32F;
        inputFormat = GL_RGB;
    }
    else if(numComponents == 4)
    {
        internalFormat = GL_RGBA32F;
        inputFormat = GL_RGBA;
    }
    else
    {
        logERROR("Texture") << "numComponents needs to be between 1 and 4!";
        logFlush();
        throw std::logic_error("numComponents needs to be between 1 and 4!");
    }

    auto tex = std::make_unique<Texture>(TextureTypes::texture2D, internalFormat, levels, width, height);
    tex->uploadData(pixels.get(), inputFormat, GL_FLOAT);

    if(generateMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeTextureFromData(const unsigned char* data, int length, int numComponents, bool generateMipmaps)
{
    int width;
    int height;
    auto pixels = loadImageData(data, length, width, height, numComponents);

    int levels = 1;
    if(generateMipmaps)
        levels = Texture::maxMipmaps(width, height, 1);

    GLenum internalFormat = 0;
    GLenum inputFormat = 0;
    if(numComponents == 1)
    {
        internalFormat = GL_R8;
        inputFormat = GL_RED;
    }
    else if(numComponents == 2)
    {
        internalFormat = GL_RG8;
        inputFormat = GL_RG;
    }
    else if(numComponents == 3)
    {
        internalFormat = GL_RGB8;
        inputFormat = GL_RGB;
    }
    else if(numComponents == 4)
    {
        internalFormat = GL_RGBA8;
        inputFormat = GL_RGBA;
    }
    else
    {
        logERROR("Texture") << "numComponents needs to be between 1 and 4!";
        logFlush();
        throw std::logic_error("numComponents needs to be between 1 and 4!");
    }

    auto tex = std::make_unique<Texture>(TextureTypes::texture2D, internalFormat, levels, width, height);
    tex->uploadData(pixels.get(), inputFormat, GL_UNSIGNED_BYTE);

    if(generateMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeTextureFromDataHDR(const unsigned char* data, int length, int numComponents, bool generateMipmaps)
{
    int width;
    int height;
    auto pixels = loadImageDataHDR(data, length, width, height, numComponents);

    int levels = 1;
    if(generateMipmaps)
        levels = Texture::maxMipmaps(width, height, 1);

    GLenum internalFormat = 0;
    GLenum inputFormat = 0;
    if(numComponents == 1)
    {
        internalFormat = GL_R32F;
        inputFormat = GL_RED;
    }
    else if(numComponents == 2)
    {
        internalFormat = GL_RG32F;
        inputFormat = GL_RG;
    }
    else if(numComponents == 3)
    {
        internalFormat = GL_RGB32F;
        inputFormat = GL_RGB;
    }
    else if(numComponents == 4)
    {
        internalFormat = GL_RGBA32F;
        inputFormat = GL_RGBA;
    }
    else
    {
        logERROR("Texture") << "numComponents needs to be between 1 and 4!";
        logFlush();
        throw std::logic_error("numComponents needs to be between 1 and 4!");
    }

    auto tex = std::make_unique<Texture>(TextureTypes::texture2D, internalFormat, levels, width, height);
    tex->uploadData(pixels.get(), inputFormat, GL_FLOAT);

    if(generateMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeTexture1DArray(int size, int numberOfLayers, GLenum internalFormat, const void* data, GLenum format, GLenum type,
                   int levels, bool genMipmaps)
{
    if(levels == 0)
        levels = Texture::maxMipmaps(size, 1, 1);

    auto tex = std::make_unique<Texture>(TextureTypes::texture1DArray, internalFormat, levels, size, numberOfLayers);

    if(data)
        tex->uploadData(data, format, type);

    if(genMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeTexture2DArray(int width, int height, int numberOfLayers, GLenum internalFormat, const void* data, GLenum format,
                   GLenum type, int levels, bool genMipmaps)
{
    if(levels == 0)
        levels = Texture::maxMipmaps(width, height, numberOfLayers);

    auto tex = std::make_unique<Texture>(TextureTypes::texture2DArray, internalFormat, levels, width, height, numberOfLayers);

    if(data)
        tex->uploadData(data, format, type);

    if(genMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeTexture2DArrayFromFiles(const std::vector<std::string>& files, int numComponents, bool generateMipmaps)
{
    int width;
    int height;
    auto pixels = loadImageFile(files[0], width, height, numComponents);

    int levels = 1;
    if(generateMipmaps)
        levels = Texture::maxMipmaps(width, height, 1);

    GLenum internalFormat = 0;
    GLenum inputFormat = 0;
    if(numComponents == 1)
    {
        internalFormat = GL_R8;
        inputFormat = GL_RED;
    }
    else if(numComponents == 2)
    {
        internalFormat = GL_RG8;
        inputFormat = GL_RG;
    }
    else if(numComponents == 3)
    {
        internalFormat = GL_RGB8;
        inputFormat = GL_RGB;
    }
    else if(numComponents == 4)
    {
        internalFormat = GL_RGBA8;
        inputFormat = GL_RGBA;
    }
    else
    {
        logERROR("Texture") << "numComponents needs to be between 1 and 4!";
        logFlush();
        throw std::logic_error("numComponents needs to be between 1 and 4!");
    }

    auto tex = std::make_unique<Texture>(TextureTypes::texture2DArray, internalFormat, levels, width, height, files.size());
    tex->upload2DLayer(pixels.get(), 0, inputFormat, GL_UNSIGNED_BYTE);

    for(size_t i=1; i < files.size(); ++i)
    {
        int new_width;
        int new_height;
        pixels = loadImageFile(files[0], new_width, new_height, numComponents);

        if(new_width != width || new_height != height)
        {
            logERROR("Texture") << "Textures in a texture array need to have the same size!";
            logFlush();
            throw std::logic_error("Textures in a texture array need to have the same size!");
        }

        tex->upload2DLayer(pixels.get(), i, inputFormat, GL_UNSIGNED_BYTE);
    }

    if(generateMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeTexture2DArrayFromFilesHDR(const std::vector<std::string>& files, int numComponents, bool generateMipmaps)
{
    int width;
    int height;
    auto pixels = loadImageFileHDR(files[0], width, height, numComponents);

    int levels = 1;
    if(generateMipmaps)
        levels = Texture::maxMipmaps(width, height, 1);

    GLenum internalFormat = 0;
    GLenum inputFormat = 0;
    if(numComponents == 1)
    {
        internalFormat = GL_R32F;
        inputFormat = GL_RED;
    }
    else if(numComponents == 2)
    {
        internalFormat = GL_RG32F;
        inputFormat = GL_RG;
    }
    else if(numComponents == 3)
    {
        internalFormat = GL_RGB32F;
        inputFormat = GL_RGB;
    }
    else if(numComponents == 4)
    {
        internalFormat = GL_RGBA32F;
        inputFormat = GL_RGBA;
    }
    else
    {
        logERROR("Texture") << "numComponents needs to be between 1 and 4!";
        logFlush();
        throw std::logic_error("numComponents needs to be between 1 and 4!");
    }

    auto tex = std::make_unique<Texture>(TextureTypes::texture2DArray, internalFormat, levels, width, height, files.size());
    tex->upload2DLayer(pixels.get(), 0, inputFormat, GL_FLOAT);

    for(int i=1; i < files.size(); ++i)
    {
        int new_width;
        int new_height;
        pixels = loadImageFileHDR(files[0], new_width, new_height, numComponents);

        if(new_width != width || new_height != height)
        {
            logERROR("Texture") << "Textures in a texture array need to have the same size!";
            logFlush();
            throw std::logic_error("Textures in a texture array need to have the same size!");
        }

        tex->upload2DLayer(pixels.get(), i, inputFormat, GL_FLOAT);
    }

    if(generateMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeCubemap(int size, GLenum internalFormat, const void* data, GLenum format, GLenum type, int levels, bool genMipmaps)
{
    if(levels == 0)
        levels = Texture::maxMipmaps(size, size, 1);

    auto tex = std::make_unique<Texture>(TextureTypes::cubemap, internalFormat, levels, size, size);

    if(data)
        tex->uploadData(data, format, type);

    if(genMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeCubemapFromFiles(const std::vector<std::string>& files, int numComponents, bool generateMipmaps)
{
    int width;
    int height;
    auto pixels = loadImageFile(files[0], width, height, numComponents);

    int levels = 1;
    if(generateMipmaps)
        levels = Texture::maxMipmaps(width, height, 1);

    GLenum internalFormat = 0;
    GLenum inputFormat = 0;
    if(numComponents == 1)
    {
        internalFormat = GL_R8;
        inputFormat = GL_RED;
    } else if(numComponents == 2)
    {
        internalFormat = GL_RG8;
        inputFormat = GL_RG;
    } else if(numComponents == 3)
    {
        internalFormat = GL_RGB8;
        inputFormat = GL_RGB;
    } else if(numComponents == 4)
    {
        internalFormat = GL_RGBA8;
        inputFormat = GL_RGBA;
    } else
    {
        logERROR("Texture") << "numComponents needs to be between 1 and 4!";
        logFlush();
        throw std::logic_error("numComponents needs to be between 1 and 4!");
    }

    auto tex = std::make_unique<Texture>(TextureTypes::cubemap, internalFormat, levels, width, height, files.size());
    tex->upload2DLayer(pixels.get(), 0, inputFormat, GL_UNSIGNED_BYTE);

    for(size_t i = 1; i < files.size(); ++i)
    {
        int new_width;
        int new_height;
        pixels = loadImageFile(files[0], new_width, new_height, numComponents);

        if(new_width != width || new_height != height)
        {
            logERROR("Texture") << "Textures in a cubemap need to have the same size!";
            logFlush();
            throw std::logic_error("Textures in a cubemap need to have the same size!");
        }

        tex->upload2DLayer(pixels.get(), i, internalFormat, GL_UNSIGNED_BYTE);
    }

    if(generateMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

std::unique_ptr<Texture>
makeCubemapFromFilesHDR(const std::vector<std::string>& files, int numComponents, bool generateMipmaps)
{
    int width;
    int height;
    auto pixels = loadImageFileHDR(files[0], width, height, numComponents);

    int levels = 1;
    if(generateMipmaps)
        levels = Texture::maxMipmaps(width, height, 1);

    GLenum internalFormat = 0;
    GLenum inputFormat = 0;
    if(numComponents == 1)
    {
        internalFormat = GL_R32F;
        inputFormat = GL_RED;
    }
    else if(numComponents == 2)
    {
        internalFormat = GL_RG32F;
        inputFormat = GL_RG;
    }
    else if(numComponents == 3)
    {
        internalFormat = GL_RGB32F;
        inputFormat = GL_RGB;
    }
    else if(numComponents == 4)
    {
        internalFormat = GL_RGBA32F;
        inputFormat = GL_RGBA;
    }
    else
    {
        logERROR("Texture") << "numComponents needs to be between 1 and 4!";
        logFlush();
        throw std::logic_error("numComponents needs to be between 1 and 4!");
    }

    auto tex = std::make_unique<Texture>(TextureTypes::cubemap, internalFormat, levels, width, height, files.size());
    tex->upload2DLayer(pixels.get(), 0, inputFormat, GL_FLOAT);

    for(size_t i=1; i < files.size(); ++i)
    {
        int new_width;
        int new_height;
        pixels = loadImageFileHDR(files[0], new_width, new_height, numComponents);

        if(new_width != width || new_height != height)
        {
            logERROR("Texture") << "Textures in a cubemap need to have the same size!";
            logFlush();
            throw std::logic_error("Textures in a cubemap need to have the same size!");
        }

        tex->upload2DLayer(pixels.get(), i, inputFormat, GL_FLOAT);
    }

    if(generateMipmaps)
        tex->generateMipmaps();

    return std::move(tex);
}

}
}