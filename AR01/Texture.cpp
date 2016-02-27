#include "Texture.h"
#include "opencv2\core.hpp"
#include <cassert>

Texture::Texture(const char* fileName)
{
	//initializating the texture mapping
	glGenTextures(1, &(this->id));
	glBindTexture(GL_TEXTURE_2D, this->id);
	glEnable(GL_TEXTURE_2D);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	cv::Mat textureImage = cv::imread(fileName);
	flip(textureImage, textureImage, -1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureImage.cols, textureImage.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, textureImage.data);
}

void Texture::Bind(unsigned int unit) {
	assert(unit >= 0 && unit <= 31);

	glActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D, this->id);
	glEnable(GL_TEXTURE_2D);


}


Texture::~Texture()
{
}
