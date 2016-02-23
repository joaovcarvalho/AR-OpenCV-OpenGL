uniform sampler2D fbo_texture;
varying vec2 f_texcoord;

void main(void) {
	vec2 uv = f_texcoord;
	vec4 original = texture2D(fbo_texture, uv);
	vec4 c = texture2D(fbo_texture, uv);

	c += texture2D(fbo_texture, uv + 0.001);
	c += texture2D(fbo_texture, uv + 0.003);
	c += texture2D(fbo_texture, uv + 0.005);
	c += texture2D(fbo_texture, uv + 0.007);
	c += texture2D(fbo_texture, uv + 0.009);
	c += texture2D(fbo_texture, uv + 0.011);

	c += texture2D(fbo_texture, uv - 0.001);
	c += texture2D(fbo_texture, uv - 0.003);
	c += texture2D(fbo_texture, uv - 0.005);
	c += texture2D(fbo_texture, uv - 0.007);
	c += texture2D(fbo_texture, uv - 0.009);
	c += texture2D(fbo_texture, uv - 0.011);

	c = c / 12;
	gl_FragColor = original;
}
