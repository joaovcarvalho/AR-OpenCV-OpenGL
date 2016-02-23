#version 400

varying vec3 TexCoord;

uniform samplerCube cube_texture;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

out vec4 fragment_colour; // final colour of surface

void main(){
	vec4 t = texture(cube_texture, TexCoord);
	fragment_colour = t; 
}
