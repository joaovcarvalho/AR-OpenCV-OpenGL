#version 330

varying vec2 TexCoord;
uniform sampler2D diffuse;

out vec4 fragment_colour; // final colour of surface

void main(){
	fragment_colour = texture2D(diffuse, TexCoord);
}
