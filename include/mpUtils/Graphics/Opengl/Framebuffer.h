/*
 * gpulic
 * Framebuffer.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Framebuffer class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

#ifndef GPULIC_FRAMEBUFFER_H
#define GPULIC_FRAMEBUFFER_H

// includes
//--------------------
#include <GL/glew.h>
#include "Texture.h"
#include <vector>
#include <map>
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

//-------------------------------------------------------------------
/**
 * class Framebuffer
 *
 * usage: Use a Frame Buffer Object to render.
 * You can attach self managed textures, or use a Renderbuffer which will be
 * managed automatically by the Framebuffer class.
 *
 */
class Framebuffer
{
public:
    Framebuffer();
    ~Framebuffer();

    operator uint32_t() const; //!< convert to opengl handle for use in raw opengl functions

    // make sampler non copyable but movable
    Framebuffer(const Framebuffer& other) = delete;
    Framebuffer& operator=(const Framebuffer& other) = delete;
    Framebuffer(Framebuffer&& other) noexcept : m_fbHandle(0){*this = std::move(other);};
    Framebuffer& operator=(Framebuffer&& other) noexcept;

    /**
     * @brief creates a renderbuffer and attaches it to the fbo
     * @param attachment the attachment point to add the attachment, eg GL_COLOR_ATTACHMENT0 or GL_DEPTH_STENCIL_ATTACHMENT
     * @param internal_format the internal format of the buffer, eg GL_RGB or GL_DEPTH24_STENCIL8
     * @param size the dsired size of the renderbuffer
     * @param samples number of samples when using supersampling
     */
    void useRenderbuffer(GLenum attachment, GLenum internal_format, glm::ivec2 size, int samples = 0);

    /**
     * @brief Attach a texture to the fbo. You remain in full responsibility for the textures livetime
     * @param attachment the attachment point to add the attachment, eg GL_COLOR_ATTACHMENT0 or GL_DEPTH_STENCIL_ATTACHMENT
     * @param texture the texture to attach to the fbo
     * @param level the texture mipmap level to attach
     */
    void attach(GLenum attachment, const Texture& texture, int level = 0);
    void detatch(GLenum attachment); //!< detatch the texture at the attachment point, will also remove a renderbuffer from the attatchment if one was added
    /**
     * @brief start using the fbo
     * @param target GL_DRAW_FRAMEBUFFER, GL_READ_FRAMEBUFFER, GL_FRAMEBUFFER
     */
    void use(GLenum target = GL_FRAMEBUFFER);
    void disable(); //!< stop using the fbo (returns to the default framebuffer)

private:
    uint32_t m_fbHandle;

    class Renderbuffer
    {
    public:
        explicit Renderbuffer(nullptr_t) : m_rbHandle(0) {}
        Renderbuffer() : m_rbHandle(0) {glCreateRenderbuffers(1,&m_rbHandle);}
        ~Renderbuffer() {glDeleteRenderbuffers(1,&m_rbHandle);}

        operator uint32_t() const {return m_rbHandle;} //!< convert to opengl handle for use in raw opengl functions

        // make sampler non copyable but movable
        Renderbuffer(const Renderbuffer& other) = delete;
        Renderbuffer& operator=(const Renderbuffer& other) = delete;
        Renderbuffer(Renderbuffer&& other) noexcept : m_rbHandle(0){*this = std::move(other);};
        Renderbuffer& operator=(Renderbuffer&& other) noexcept {using std::swap; swap(m_rbHandle,other.m_rbHandle); return *this;};

    private:
        uint32_t m_rbHandle;
    };

    std::map<GLenum, Renderbuffer> m_color_renderbuffers;
    Renderbuffer m_depth_stencil_renderbuffer{nullptr};

    std::vector<GLenum> m_draw_buffers;
    GLenum m_currentTarget{0};
};

}}

#endif //GPULIC_FRAMEBUFFER_H
