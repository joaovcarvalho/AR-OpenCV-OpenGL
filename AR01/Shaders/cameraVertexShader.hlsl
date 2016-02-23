#version 330

in vec3 vertex_position;
in vec2 vertex_texture;

varying vec2 TexCoord;

void main(){
  	TexCoord = vertex_texture;
	gl_Position = vec4(vertex_position, 1.0);
}


  
