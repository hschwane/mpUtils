/*
 * mpUtils
 * Shader.cpp
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

#include "mpUtils/Graphics/Opengl/Shader.h"
#include "glm/ext.hpp"

namespace mpu {
namespace gph {

    std::vector<glsp::files::path> shader_include_paths;
    void addShaderIncludePath(glsp::files::path include_path)
    {
        shader_include_paths.emplace_back(std::forward<glsp::files::path>(include_path));
    }

	ShaderModule::ShaderModule(const ShaderStage stage, const glsp::files::path path_to_file)
		: stage(stage), path_to_file(path_to_file)
	{
	}

	ShaderModule::ShaderModule(glsp::files::path path_to_file)
		: ShaderModule([ext = path_to_file.extension()]()
	{
		if (ext == ".vert") return ShaderStage::eVertex;
		if (ext == ".frag") return ShaderStage::eFragment;
		if (ext == ".geom") return ShaderStage::eGeometry;
		if (ext == ".comp") return ShaderStage::eCompute;
		if (ext == ".tesc") return ShaderStage::eTessControl;
		if (ext == ".tese") return ShaderStage::eTessEvaluation;

        logERROR("ShaderModule") << "Wrong File extension. Should be one of .vert, .frag, .geom, .comp, .tesc or .tese.";
        throw std::runtime_error("File extension should be one of .vert, .frag, .geom, .comp, .tesc or .tese.");
	}(), path_to_file)
	{
	}

	ShaderProgram::ShaderProgram()
		: m_progHandle(0)
	{
        // init glsp debugging once
        static struct DoOnce
        {
            DoOnce() {
                glsp::ERR_OUTPUT = [](const std::string& x){ logERROR("glsp") << x;};
            }
        } doOnce;
	}

    ShaderProgram::~ShaderProgram()
    {
        if(m_progHandle != 0)
            glDeleteProgram(m_progHandle);
    }

    ShaderProgram::operator uint32_t() const
    {
        return m_progHandle;
    }

    ShaderProgram::ShaderProgram(const ShaderModule& shader, std::vector<glsp::definition> definitions)
		: ShaderProgram({shader}, std::move(definitions))
	{
	}

	ShaderProgram::ShaderProgram(std::initializer_list<const ShaderModule> shaders, std::vector<glsp::definition> definitions)
		: ShaderProgram(shaders.begin(),shaders.end(), std::move(definitions))
	{
	}

    ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept
    {
        using std::swap;
        swap(m_progHandle,other.m_progHandle);
        swap(m_shaderSources,other.m_shaderSources);
        swap(m_preprocessorDefinitions,other.m_preprocessorDefinitions);
        return *this;
    }

    void ShaderProgram::rebuild()
    {

        assert_true(!m_shaderSources.empty(), "ShaderProgram", "The ShaderProgram has no sources.");
        assert_true(m_shaderSources.count(ShaderStage::eCompute) != 1 || m_shaderSources.size() == 1, "ShaderProgram", "When using compute shaders no other shader stage is allowed.");
        assert_true(!(m_shaderSources.size() == 1 && m_shaderSources.count(ShaderStage::eCompute) == 0), "ShaderProgram", "It's not allowed to have only one Shader stage it it's not a compute shader.");
        assert_true(m_shaderSources.count(ShaderStage::eVertex) != 0 || m_shaderSources.count(ShaderStage::eCompute) == 1, "ShaderProgram", "Missing a Vertex Shader.");
        assert_true(m_shaderSources.count(ShaderStage::eFragment) != 0 || m_shaderSources.count(ShaderStage::eCompute) == 1, "ShaderProgram", "Missing a Fragment Shader.");
        assert_true(((m_shaderSources.count(ShaderStage::eTessControl) + m_shaderSources.count(ShaderStage::eTessEvaluation)) & 0x1) == 0,
                    "ShaderProgram", "When using Tesselation Shaders, you have to provide a Tesselation Control Shader as well as a Tesselation Evaluation Shader.");

        // get new shader program id
        if(m_progHandle != 0)
        {
            glDeleteProgram(m_progHandle);
            m_progHandle = 0;
        }
        m_progHandle = glCreateProgram();

        // compile all stages
        std::vector<uint32_t> handles;
        for(auto&& shaderSource : m_shaderSources)
        {
            // add the current shader stage to the definitions
            auto currentDefinitions = m_preprocessorDefinitions;
            switch(shaderSource.first)
            {
                case ShaderStage::eVertex:
                    currentDefinitions.emplace_back("__VERTEX_SHADER__");
                    break;
                case ShaderStage::eTessControl:
                    currentDefinitions.emplace_back("__TESS_CONTROL_SHADER__");
                    break;
                case ShaderStage::eTessEvaluation:
                    currentDefinitions.emplace_back("__TESS_EVAL_SHADER__");
                    break;
                case ShaderStage::eGeometry:
                    currentDefinitions.emplace_back("__GEOMETRY_SHADER__");
                    break;
                case ShaderStage::eFragment:
                    currentDefinitions.emplace_back("__FRAGMENT_SHADER__");
                    break;
                case ShaderStage::eCompute:
                    currentDefinitions.emplace_back("__COMPUTE_SHADER__");
                    break;
            }

            // first use glsp to do preprocessing
            glsp::processed_file result;
            if(shaderSource.second.first.empty() && !shaderSource.second.second.empty())
            {
                // source given as a string
                result = glsp::preprocess_source(shaderSource.second.second, "ShaderString", shader_include_paths, m_preprocessorDefinitions);

            } else if(!shaderSource.second.first.empty() && shaderSource.second.second.empty())
            {
                // source given as a file
                assert_critical(glsp::files::exists(shaderSource.second.first), "ShaderProgram",  "Shader file not found: \"" + shaderSource.second.first.string() + "\"");
                result = glsp::preprocess_file(shaderSource.second.first, shader_include_paths, m_preprocessorDefinitions);
            } else
            {
                logERROR("ShaderProgram") << "Unexpected error during shader program rebuild.";
                logFlush();
                throw std::logic_error("Program error during ShaderProgram rebuild");
            }

            if(!result.valid())
            {
                logERROR("ShaderProgram") << "Errors during preprocessing of shader " << shaderSource.second.first << ".";
                logFlush();
                throw std::runtime_error("Errors during shader compilation: preprocessor.");
            }

            // now generate a handle and compile
            int id=handles.size();
            handles.push_back(glCreateShader(static_cast<GLenum>(shaderSource.first)));
            const char *c_str = result.contents.c_str();
            glShaderSource(handles[id], 1, &c_str, nullptr);
            glCompileShader(handles[id]);
            {
                int success;
                glGetShaderiv(handles[id], GL_COMPILE_STATUS, &success);
                if (!success)
                {
                    int log_length;
                    glGetShaderiv(handles[id], GL_INFO_LOG_LENGTH, &log_length);
                    std::string log(log_length, ' ');
                    glGetShaderInfoLog(handles[id], log_length, &log_length, &log[0]);

                    logERROR("ShaderProgram") << "Compiling shader " << shaderSource.second.first << " failed. Compiler Log: " << log;
                    logFlush();
                    throw std::runtime_error("Shader compilation failed.");
                }
            }

            glAttachShader(*this, handles[id]);
        }

        // now link
        glLinkProgram(*this);
		{
			int success;
			glGetProgramiv(*this, GL_LINK_STATUS, &success);
			if(!success)
			{
				int log_length;
				glGetProgramiv(*this, GL_INFO_LOG_LENGTH, &log_length);
				std::string log(log_length, ' ');
				glGetProgramInfoLog(*this, log_length, &log_length, &log[0]);

                logERROR("ShaderProgram") << "Linking shader failed. Linker Log: " << log;
                logFlush();
                throw std::runtime_error("Linking shader failed.");
			}
		}

		for (const auto& handle : handles)
        {
            glDetachShader(*this, handle);
            glDeleteShader(handle);
        }
    }

    void ShaderProgram::setShaderModule(ShaderModule module)
    {
        m_shaderSources[module.stage] = std::make_pair(module.path_to_file,"");
    }

    void ShaderProgram::setShaderSource(ShaderStage stage, std::string source)
    {
        m_shaderSources[stage] = std::make_pair(glsp::files::path(),source);
    }

    void ShaderProgram::removeShaderStage(ShaderStage stage)
    {
        m_shaderSources.erase(stage);
    }

    void ShaderProgram::addDefinition(glsp::definition def)
        {
            m_preprocessorDefinitions.push_back(std::move(def));
        }

    void ShaderProgram::clearDefinitions()
    {
        m_preprocessorDefinitions.clear();
    }

    int ShaderProgram::attributeLocation(const std::string& attribute) const
	{
		return glGetProgramResourceLocation(*this, GL_PROGRAM_INPUT, attribute.data());
	}

	int ShaderProgram::uniformLocation(const std::string& uniform) const
	{
		return glGetProgramResourceLocation(*this, GL_UNIFORM, uniform.data());
	}

	int ShaderProgram::uniformBlock(const std::string& uniform) const
	{
		return glGetUniformBlockIndex(*this, uniform.data());
	}

	void ShaderProgram::use() const
	{
		glUseProgram(*this);
	}

	void ShaderProgram::dispatch(const uint32_t invocations, const uint32_t group_size) const
	{
		dispatch(glm::uvec3(invocations, 1, 1), glm::uvec3(group_size, 1, 1));
	}

	void ShaderProgram::dispatch(const glm::uvec2 invocations, const glm::uvec2 group_size) const
	{
		dispatch(glm::uvec3(invocations.x, invocations.y, 1), glm::uvec3(group_size.x, group_size.y, 1));
	}

	void ShaderProgram::dispatch(const glm::uvec3 invocations, const glm::uvec3 group_size) const
	{
		const static auto invocation_count = [](uint32_t global, uint32_t local)
		{
			return (global % local == 0) ? global / local : global / local + 1;
		};

		use();
		glDispatchComputeGroupSizeARB(
			invocation_count(invocations.x, group_size.x),
			invocation_count(invocations.y, group_size.y),
			invocation_count(invocations.z, group_size.z),
			group_size.x,
			group_size.y,
			group_size.z
		);
	}

	void ShaderProgram::dispatch(uint32_t groups) const
	{
		use();
		glDispatchCompute(groups,1,1);
	}

	void ShaderProgram::dispatch(glm::u32vec2 groups) const
	{
		use();
		glDispatchCompute(groups.x,groups.y,1);
	}

	void ShaderProgram::dispatch(glm::uvec3 groups) const
	{
		use();
		glDispatchCompute(groups.x,groups.y,groups.z);
	}

	void ShaderProgram::uniform1i(const std::string& uniform, const int32_t value) const
	{
		glProgramUniform1i(*this, uniformLocation(uniform), value);
	}

    void ShaderProgram::uniform2i(const std::string& uniform, const glm::ivec2& value) const
    {
        glProgramUniform2iv(*this, uniformLocation(uniform), 1, glm::value_ptr(value));
    }

    void ShaderProgram::uniform3i(const std::string& uniform, const glm::ivec3& value) const
    {
        glProgramUniform3iv(*this, uniformLocation(uniform), 1, glm::value_ptr(value));
    }

    void ShaderProgram::uniform4i(const std::string& uniform, const glm::ivec4& value) const
    {
        glProgramUniform4iv(*this, uniformLocation(uniform), 1, glm::value_ptr(value));
    }

    void ShaderProgram::uniform1ui(const std::string& uniform, const uint32_t value) const
    {
        glProgramUniform1ui(*this, uniformLocation(uniform), value);
    }

    void ShaderProgram::uniform2ui(const std::string& uniform, const glm::uvec2& value) const
    {
        glProgramUniform2uiv(*this, uniformLocation(uniform), 1, glm::value_ptr(value));
    }

    void ShaderProgram::uniform3ui(const std::string& uniform, const glm::uvec3& value) const
    {
        glProgramUniform3uiv(*this, uniformLocation(uniform), 1, glm::value_ptr(value));
    }

    void ShaderProgram::uniform4ui(const std::string& uniform, const glm::uvec4& value) const
    {
        glProgramUniform4uiv(*this, uniformLocation(uniform), 1, glm::value_ptr(value));
    }

    void ShaderProgram::uniform1f(const std::string& uniform, const float value) const
	{
        glProgramUniform1f(*this, uniformLocation(uniform), value);
	}

    void ShaderProgram::uniform2f(const std::string& uniform, const glm::vec2& vec) const
    {
        glProgramUniform2fv(*this, uniformLocation(uniform), 1, glm::value_ptr(vec));
    }

    void ShaderProgram::uniform3f(const std::string& uniform, const glm::vec3& vec) const
    {
        glProgramUniform3fv(*this, uniformLocation(uniform), 1, glm::value_ptr(vec));
    }

    void ShaderProgram::uniform4f(const std::string& uniform, const glm::vec4& vec) const
    {
        glProgramUniform4fv(*this, uniformLocation(uniform), 1, glm::value_ptr(vec));
    }

    void ShaderProgram::uniformMat2(const std::string& uniform, const glm::mat2& mat, const bool transpose) const
    {
        glProgramUniformMatrix2fv(*this, uniformLocation(uniform), 1, transpose, glm::value_ptr(mat));
    }

    void ShaderProgram::uniformMat3(const std::string& uniform, const glm::mat3& mat, const bool transpose) const
    {
        glProgramUniformMatrix3fv(*this, uniformLocation(uniform), 1, transpose, glm::value_ptr(mat));
    }

    void ShaderProgram::uniformMat4(const std::string& uniform, const glm::mat4& mat, const bool transpose) const
    {
        glProgramUniformMatrix4fv(*this, uniformLocation(uniform), 1, transpose, glm::value_ptr(mat));
    }

    void ShaderProgram::uniformMat2x3(const std::string& uniform, const glm::mat2x3& mat, const bool transpose) const
    {
        glProgramUniformMatrix2x3fv(*this, uniformLocation(uniform), 1, transpose, glm::value_ptr(mat));
    }

    void ShaderProgram::uniformMat3x2(const std::string& uniform, const glm::mat3x2& mat, const bool transpose) const
    {
        glProgramUniformMatrix3x2fv(*this, uniformLocation(uniform), 1, transpose, glm::value_ptr(mat));
    }

    void ShaderProgram::uniformMat2x4(const std::string& uniform, const glm::mat2x4& mat, const bool transpose) const
    {
        glProgramUniformMatrix2x4fv(*this, uniformLocation(uniform), 1, transpose, glm::value_ptr(mat));
    }

    void ShaderProgram::uniformMat4x2(const std::string& uniform, const glm::mat4x2& mat, const bool transpose) const
    {
        glProgramUniformMatrix4x2fv(*this, uniformLocation(uniform), 1, transpose, glm::value_ptr(mat));
    }

    void ShaderProgram::uniformMat4x3(const std::string& uniform, const glm::mat4x3& mat, const bool transpose) const
    {
        glProgramUniformMatrix4x3fv(*this, uniformLocation(uniform), 1, transpose, glm::value_ptr(mat));
    }

    void ShaderProgram::uniformMat3x4(const std::string& uniform, const glm::mat3x4& mat, const bool transpose) const
    {
        glProgramUniformMatrix3x4fv(*this, uniformLocation(uniform), 1, transpose, glm::value_ptr(mat));
    }

    void ShaderProgram::uniform1b(const std::string& uniform, const bool value) const
    {
        uniform1i(uniform, static_cast<int32_t>(value));
    }

    void ShaderProgram::uniform1ui64(const std::string& uniform, const uint64_t value) const
    {
        glProgramUniform1ui64ARB(*this, uniformLocation(uniform), value);
    }

}}