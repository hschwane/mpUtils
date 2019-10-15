/*
 * gpulic
 * Framebuffer.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Implements the Framebuffer class
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */

// includes
//--------------------
#include "mpUtils/Log/Log.h"
#include "mpUtils/Graphics/Opengl/Framebuffer.h"
//--------------------

// namespace
//--------------------
namespace mpu {
namespace gph {
//--------------------

// function definitions of the Framebuffer class
//-------------------------------------------------------------------
Framebuffer::Framebuffer() : m_fbHandle(0)
{
    glCreateFramebuffers(1,&m_fbHandle);
}

Framebuffer::~Framebuffer()
{
    glDeleteFramebuffers(1,&m_fbHandle);
}

Framebuffer& Framebuffer::operator=(Framebuffer&& other) noexcept
{
    using std::swap;
    swap(m_fbHandle,other.m_fbHandle);
    swap(m_color_renderbuffers,other.m_color_renderbuffers);
    swap(m_depth_stencil_renderbuffer,other.m_depth_stencil_renderbuffer);
    swap(m_draw_buffers,other.m_draw_buffers);
    return *this;
}

Framebuffer::operator uint32_t() const
{
    return m_fbHandle;
}

void Framebuffer::useRenderbuffer(const GLenum attachment, const GLenum internal_format, const glm::ivec2 size, int samples)
{
    switch (attachment)
    {
        case GL_DEPTH_STENCIL_ATTACHMENT:
            m_depth_stencil_renderbuffer = Renderbuffer();
            glNamedRenderbufferStorageMultisample(m_depth_stencil_renderbuffer, samples, internal_format, size.x, size.y);
            glNamedFramebufferRenderbuffer(*this, attachment, GL_RENDERBUFFER, m_depth_stencil_renderbuffer);
            break;
        case GL_DEPTH_ATTACHMENT:
        case GL_STENCIL_ATTACHMENT:
            assert_true(false, "Framebuffer", "Invalid Attachments: Depth-only or Stencil-only attachments are not supported. Use a full GL_DEPTH_STENCIL_ATTACHMENT instead.");
            break;
        default:
            auto&& buffer = m_color_renderbuffers[attachment];
            buffer = Renderbuffer();
            glNamedRenderbufferStorageMultisample(buffer, samples, internal_format, size.x, size.y);
            glNamedFramebufferRenderbuffer(*this, attachment, GL_RENDERBUFFER, buffer);
            if (std::find(m_draw_buffers.begin(), m_draw_buffers.end(), attachment) == m_draw_buffers.end())
            {
                m_draw_buffers.push_back(attachment);
                glNamedFramebufferDrawBuffers(*this, static_cast<int>(m_draw_buffers.size()), m_draw_buffers.data());
            }
            break;
    }

    assert_true(glCheckNamedFramebufferStatus(*this, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"Framebuffer", "Framebuffer incomplete.");
}

void Framebuffer::attach(const GLenum attachment, const Texture& texture, const int level)
{
    glNamedFramebufferTexture(*this, attachment, texture, level);
    switch(attachment)
    {
        case GL_DEPTH_STENCIL_ATTACHMENT:
            m_depth_stencil_renderbuffer = Renderbuffer(nullptr);
            break;
        case GL_DEPTH_ATTACHMENT:
        case GL_STENCIL_ATTACHMENT:
            assert_true(false,"Framebuffer", "Invalid Attachments: Depth-only or Stencil-only attachments are not supported. Use a full GL_DEPTH_STENCIL_ATTACHMENT instead.");
            break;
        default:
            m_color_renderbuffers.erase(attachment);
            if (std::find(m_draw_buffers.begin(), m_draw_buffers.end(), attachment) == m_draw_buffers.end())
            {
                m_draw_buffers.push_back(attachment);
                glNamedFramebufferDrawBuffers(*this, static_cast<int>(m_draw_buffers.size()), m_draw_buffers.data());
            }
            break;
    }

    assert_true(glCheckNamedFramebufferStatus(*this, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "Framebuffer", "Framebuffer incomplete.");
}

void Framebuffer::detatch(GLenum attachment)
{
    glNamedFramebufferTexture(*this, attachment, 0, 0);
    switch(attachment)
    {
        case GL_DEPTH_STENCIL_ATTACHMENT:
            m_depth_stencil_renderbuffer = Renderbuffer(nullptr);
            break;
        case GL_DEPTH_ATTACHMENT:
        case GL_STENCIL_ATTACHMENT:
            assert_true(false,"Framebuffer", "Invalid Attachments: Depth-only or Stencil-only attachments are not supported. Use a full GL_DEPTH_STENCIL_ATTACHMENT instead.");
            break;
        default:
            m_color_renderbuffers.erase(attachment);
            if (std::find(m_draw_buffers.begin(), m_draw_buffers.end(), attachment) == m_draw_buffers.end())
            {
                m_draw_buffers.push_back(attachment);
                glNamedFramebufferDrawBuffers(*this, static_cast<int>(m_draw_buffers.size()), m_draw_buffers.data());
            }
            break;
    }
}

void Framebuffer::use(GLenum target)
{
    assert_true(glCheckNamedFramebufferStatus(*this, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"Framebuffer", "Framebuffer incomplete.");
    m_currentTarget = target;
    glBindFramebuffer(target, *this);
}

void Framebuffer::disable()
{
    if(m_currentTarget != 0)
    {
        glBindFramebuffer(m_currentTarget,0);
        m_currentTarget = 0;
    }
}


}}