#version 330

in vec3 LightIntensity;
varying vec4 position_eye, normal_eye;
varying vec2 TexCoord;
varying vec3 normal;

uniform sampler2D diffuse;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

out vec4 fragment_colour; // final colour of surface

void main(){
	fragment_colour = vec4 (1.0,1.0,1.0, 1.0);
}