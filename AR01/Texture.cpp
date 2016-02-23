#include "Texture.h"
#include <cassert>

Texture::Texture(const char* fileName)
{
}

void Texture::Bind(unsigned int unit) {
	assert(unit >= 0 && unit <= 31);

	glActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D, this->id);
}


Texture::~Texture()
{
}
