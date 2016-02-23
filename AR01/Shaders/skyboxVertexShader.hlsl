#version 400

in vec3 vertex_position;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

varying vec3 TexCoord;

void main(){
  vec4 position_eye = vec4 (view * model * vec4 (vertex_position, 1.0));
  TexCoord = vertex_position;
  gl_Position = proj * position_eye;
}


  
