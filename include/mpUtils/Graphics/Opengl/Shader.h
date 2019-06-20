/*
 * mpUtils
 * Shader.h
 *
 * Contains the Shader class which is used to manage and compile openGL shader.
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2017 Hendrik Schwanekamp
 *
 * This file was originally written and generously provided for this framework from Johannes Braun.
 *
 */

#pragma once

#include <algorithm>
#include <vector>
#include <map>
#include <iterator>
#include <GL/glew.h>
#include <glm/glm.hpp>

#include <glsp/preprocess.hpp>
#include "Handle.h"
#include "mpUtils/Log/Log.h"

namespace mpu {
namespace gph {


    extern std::vector<glsp::files::path> shader_include_paths;
    void addShaderIncludePath(glsp::files::path include_path);

	/**
	 * enum of all shader stages for type-safety
	 */
	enum class ShaderStage
	{
		eVertex = GL_VERTEX_SHADER,
		eTessControl = GL_TESS_CONTROL_SHADER,
		eTessEvaluation = GL_TESS_EVALUATION_SHADER,
		eGeometry = GL_GEOMETRY_SHADER,
		eFragment = GL_FRAGMENT_SHADER,
		eCompute = GL_COMPUTE_SHADER,
	};

	/**
	 * struct to combine a shader stage and a source file
	 */
	struct ShaderModule
	{
		ShaderModule() = default;
		ShaderModule( ShaderStage stage, glsp::files::path path_to_file);
		ShaderModule(glsp::files::path path_to_file); //!< stage is determined by file extension

		ShaderStage stage;
        glsp::files::path path_to_file;
	};

    /**
     * class ShaderProgram
     * A class to manage a shader program
     *
     * usage:
     * Use one of the constructors and pass a shaderModule struct for each shader stage you want to use. You can also provide definitions that are passed
     * to the preprocessor. The function mpu::gph::addShaderIncludePath() can be used to globally add include paths.
     * After that you can modify the shader program by using setShaderModule(), setShaderSource(), addDefinition() and clearDefinitions().
     * Use rebuild to rebuild the shader program according to the current settings.
     * You can also use rebuild() to rebuild the program from an entirely new set of shader modules.
     *
     * After that you set your uniforms by using the functions below. For arrays consider a ssto or
     * call the glProgramUniform**v() function on your own.
     * Now you can bind your shader using the use() function and start to render something.
     *
     * Compute Shader:
     * To compile a compute shader just use une of the constructors or the rebuild function as described above and then call one of the dispatch() functions.
     *
     * Preprocessor:
     * When compiling the custom c/c++ style preprocssor written by Johannes Braun is used on the shader and provides the ability to use
     * constructs like "#include", "#define", "#ifdef" and other preprocessor macros. See https://github.com/johannes-braun/GLshader.
     * Depending on the shader stage compiled, one of the following definitions will be added:
     * __VERTEX_SHADER__
     * __TESS_CONTROL_SHADER__
     * __TESS_EVAL_SHADER__
     * __GEOMETRY_SHADER__
     * __FRAGMENT_SHADER__
     * __COMPUTE_SHADER__
     *
     */
	class ShaderProgram : public Handle<uint32_t, decltype(&glCreateProgram), &glCreateProgram, decltype(&glDeleteProgram), &glDeleteProgram>
	{
	public:
		ShaderProgram();
		explicit ShaderProgram(nullptr_t);
		explicit ShaderProgram(const ShaderModule& shader, std::vector<glsp::definition> definitions = {}); //!< construct shader program from a single shader module
		ShaderProgram(std::initializer_list<const ShaderModule> shaders, std::vector<glsp::definition> definitions = {}); //!< construct shader program from multiple modules

		template<typename TIterator>
		ShaderProgram(TIterator begin, TIterator end, std::vector<glsp::definition> definitions = {}); //!< construct the shader program from some container of shader modules

		void rebuild(); //!< rebuild shader program with current settings (files will be loaded again)

		void setShaderModule(ShaderModule module); //!< replaces whatever is currently set for the shader stage module belongs to
		void setShaderSource(ShaderStage stage, std::string source); //!< set some string as the source code for the stage stage replaces the current file or string for that stage
        void removeShaderStage(ShaderStage stage); //!< removes the shader source registered for the stage "stage"

		void addDefinition(glsp::definition def);   //!< add a definition that will be used on the shaders programs next rebuild
        void clearDefinitions();    //!< remove all existing definitions

		void use() const; //!< use the compute shader for the next rendering call

		// compute shader dispatching -----------------------------------

		void dispatch(uint32_t invocations, uint32_t group_size) const; //!< start a 1D compute shader run
		void dispatch(glm::u32vec2 invocations, glm::u32vec2 group_size) const; //!< start a 2D compute shader run
		void dispatch(glm::uvec3 invocations, glm::uvec3 group_size) const; //!< start a 3D compute shader run

        void dispatch(uint32_t groups) const; //!< start a 1D compute shader run using a fixed group size
        void dispatch(glm::u32vec2 groups) const; //!< start a 2D compute shader run using a fixed group size
        void dispatch(glm::uvec3 groups) const; //!< start a 3D compute shader run using a fixed group size

        // uniforms and attribute queries ---------------------------------

        int attributeLocation(const std::string& attribute) const; //!< query a given attributes location
        int uniformLocation(const std::string& uniform) const; //!< query a given uniforms location
        int uniformBlock(const std::string& uniform) const; //!< query a given uniform blocks location

        // uniform upload functions --------------------------------------------

        // upload vector of ints
		void uniform1i(const std::string& uniform, int32_t value) const; //!< upload an integer to a uniform
		void uniform2i(const std::string& uniform, const glm::ivec2& value) const; //!< upload an integer vec2 to a uniform
		void uniform3i(const std::string& uniform, const glm::ivec3& value) const; //!< upload an integer vec3 to a uniform
		void uniform4i(const std::string& uniform, const glm::ivec4& value) const; //!< upload an integer vec4 to a uniform

        // upload vectors of unsigned ints
        void uniform1ui(const std::string& uniform, uint32_t value) const; //!< upload an unsigned integer to a uniform
        void uniform2ui(const std::string& uniform, const glm::uvec2& value) const; //!< upload an unsigned vec2 integer to a uniform
        void uniform3ui(const std::string& uniform, const glm::uvec3& value) const; //!< upload an unsigned vec3 integer to a uniform
        void uniform4ui(const std::string& uniform, const glm::uvec4& value) const; //!< upload an unsigned vec4 integer to a uniform

        // upload vector of floats
        void uniform1f(const std::string& uniform, float value) const; //!< upload a float to a uniform
        void uniform2f(const std::string& uniform, const glm::vec2& vec) const; //!< upload a float vec2 to a uniform
        void uniform3f(const std::string& uniform, const glm::vec3& vec) const; //!< upload a float vec2 to a uniform
        void uniform4f(const std::string& uniform, const glm::vec4& vec) const; //!< upload a float vec2 to a uniform

        // upload matrices
		void uniformMat2(const std::string& uniform, const glm::mat2& mat, bool transpose=false) const; //!< upload a mat2 to a uniform
		void uniformMat3(const std::string& uniform, const glm::mat3& mat, bool transpose=false) const; //!< upload a mat3 to a uniform
		void uniformMat4(const std::string& uniform, const glm::mat4& mat, bool transpose=false) const; //!< upload a mat4 to a uniform
		void uniformMat2x3(const std::string& uniform, const glm::mat2x3& mat, bool transpose=false) const; //!< upload a mat2x3 to a uniform
		void uniformMat3x2(const std::string& uniform, const glm::mat3x2& mat, bool transpose=false) const; //!< upload a mat3x2 to a uniform
		void uniformMat2x4(const std::string& uniform, const glm::mat2x4& mat, bool transpose=false) const; //!< upload a mat2x4 to a uniform
		void uniformMat4x2(const std::string& uniform, const glm::mat4x2& mat, bool transpose=false) const; //!< upload a mat4x2 to a uniform
		void uniformMat4x3(const std::string& uniform, const glm::mat4x3& mat, bool transpose=false) const; //!< upload a mat4x2 to a uniform
		void uniformMat3x4(const std::string& uniform, const glm::mat3x4& mat, bool transpose=false) const; //!< upload a mat4x2 to a uniform

        // special
        void uniform1b(const std::string& uniform, bool value) const; //!< upload a boolean to a uniform
        void uniform1ui64(const std::string& uniform, uint64_t value) const; //!< upload a 64bit unsigned int to a uniform

	private:

	    std::map<ShaderStage, std::pair<glsp::files::path,std::string>> m_shaderSources;
        std::vector<glsp::definition> m_preprocessorDefinitions;

		// Only used temporarily when constructing the ShaderProgram.
		using ShaderHandle = Handle<uint32_t, decltype(&glCreateShader), &glCreateShader, decltype(&glDeleteShader), &glDeleteShader, GLenum>;
	};


//-------------------------------------------------------------------
// definitions of template functions of the ShaderProgram class

template <typename TIterator>
ShaderProgram::ShaderProgram(TIterator begin, TIterator end, std::vector<glsp::definition> definitions)
    : ShaderProgram()
{
    static_assert(std::is_same<std::decay_t<typename std::iterator_traits<TIterator>::value_type>, ShaderModule>::value,
                  "This constructor only accepts iterators of containers containing ShaderModule objects.");

    m_preprocessorDefinitions = std::move(definitions);
    std::for_each(begin, end, [this](const ShaderModule& shader)
    {
        assert_true(this->m_shaderSources.count(shader.stage) == 0, "ShaderProgram", "You cannot have multiple Shader modules with the same stage in the same program.");
        this->m_shaderSources[shader.stage] = std::make_pair(shader.path_to_file,"");
    });

    rebuild();
}

}}