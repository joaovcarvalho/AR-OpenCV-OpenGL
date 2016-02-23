#version 330

in vec3 LightIntensity;
varying vec4 position_eye, normal_eye;
varying vec2 TexCoord;
varying vec3 normal;

uniform sampler2D diffuse;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

vec3 light_position_world = ( vec3 (0.0, 0.0, 0.0));
vec3 Ls = vec3 (1.0, 1.0, 1.0); // white specular colour
vec3 Ld = vec3 (0.9, 0.9, 0.9); // dull white diffuse light colour
vec3 La = vec3 (0.2, 0.2, 0.2); // grey ambient colour

// surface reflectance
vec3 Ks = vec3 (0.1, 0.1, 0.1); // fully reflect specular light
vec3 Kd = vec3 (1.0, 1.0, 1.0); // diffuse surface reflectance
vec3 Ka = vec3 (0.0, 0.0, 0.0); // fully reflect ambient light

float specular_exponent = 500.0; // specular 'power'
out vec4 fragment_colour; // final colour of surface

void main(){
	vec3 Ia = La*Ka;
	Ld = texture2D(diffuse, TexCoord).xyz;
	
	vec3 light_position_eye = vec3(view * vec4(light_position_world,1.0));
	vec4 distance_to_light_eye = light_position_eye - position_eye;
	vec3 direction_to_light_eye = normalize(  vec3(distance_to_light_eye) );

	float dot_prod = dot (direction_to_light_eye, vec3(normal_eye));
	dot_prod = max (dot_prod, 0.0);
	vec3 Id = Ld * Kd * dot_prod; 

	vec3 reflection_eye = reflect(-direction_to_light_eye, vec3(normal_eye));
	vec3 surface_to_viewer_eye = normalize(-vec3(position_eye));
	float dot_prod_specular = dot(reflection_eye,surface_to_viewer_eye);
	dot_prod_specular = max(dot_prod_specular, 0.0);
	float specular_factor = pow(dot_prod_specular, specular_exponent);
	vec3 Is = Ls*Ks*specular_factor;

	fragment_colour = vec4 (Id + Is + Ia, 1.0);
}