#pragma once

#include "stdafx.h"
class Texture
{
private:
	GLuint id;

public:
	Texture(const char* fileName);
	~Texture();

	void Bind(unsigned int unit);
};

