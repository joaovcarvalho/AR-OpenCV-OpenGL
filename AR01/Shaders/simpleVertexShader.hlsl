#version 330

in vec3 vertex_position;
in vec3 vertex_normal;
in vec2 vertex_texture;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

varying vec3 normal;
varying vec4 position_eye, normal_eye;
varying vec2 TexCoord;

void main(){
  normal = vertex_normal;
  position_eye = vec4 (view * model * vec4 (vertex_position, 1.0));
  normal_eye = vec4 (view * model * vec4 (vertex_normal, 0.0));
  TexCoord = vertex_texture;
  gl_Position = proj * position_eye;
}


  
