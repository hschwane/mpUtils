/*
 * mpUtils
 * mpGraphics.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2018 Hendrik Schwanekamp
 *
 */
#ifndef MPUTILS_MPGRAPHICS_H
#define MPUTILS_MPGRAPHICS_H

// external includes
//--------------------
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
//--------------------

// path for shader and resources
//--------------------
#if !defined(MPU_NO_PATHS)
    #include "mpUtils/paths.h"
#endif
//--------------------

// framework graphics stuff
//--------------------
#include "mpUtils/Graphics/Window.h"
#include "mpUtils/Graphics/Utils/Transform.h"
#include "mpUtils/Graphics/Utils/Transform2D.h"
#include "mpUtils/Graphics/Utils/misc.h"
#include "mpUtils/Graphics/Opengl/Buffer.h"
#include "mpUtils/Graphics/Opengl/VertexArray.h"
#include "mpUtils/Graphics/Opengl/Shader.h"
#include "mpUtils/Graphics/Opengl/Texture.h"
#include "mpUtils/Graphics/Opengl/Framebuffer.h"
#include "mpUtils/Graphics/Opengl/Sampler.h"
#include "mpUtils/Graphics/Opengl/SyncObject.h"
#include "mpUtils/Graphics/Rendering/ScreenFillingTri.h"
#include "mpUtils/Graphics/Rendering/Camera.h"
#include "mpUtils/Graphics/Rendering2D/Sprite2D.h"
#include "mpUtils/Graphics/Rendering2D/Renderer2D.h"
#include "mpUtils/Graphics/Rendering2D/Camera2D.h"
#include "mpUtils/Graphics/Gui/ImGui.h"
#include "mpUtils/Graphics/Gui/ImGuiStyles.h"
#include "mpUtils/Graphics/Gui/ImGuiElements.h"
#include "mpUtils/Graphics/Input.h"
#include "mpUtils/Graphics/Scenes/Scene.h"
#include "mpUtils/Graphics/Scenes/SceneManager.h"
//--------------------

#endif //MPUTILS_MPGRAPHICS_H
