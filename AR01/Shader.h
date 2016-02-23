#pragma once

#include "stdafx.h"

class Shader
{
private:
	GLuint id;
	char* vertexShaderFile;
	char* fragmentShaderFile;

public:
	Shader(char*, char*);
	~Shader();

	char* readShaderSource(const char* shaderFile);
	void AddShader(GLuint ShaderProgram, const char* pShaderText, GLenum ShaderType);
	void CompileShaders();

	GLuint getProgramID();
};

