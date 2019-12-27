#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(std140, binding = 1) buffer src {
	vec4 AB[];	// A and B concentration.
};

layout(binding = 2) uniform Parameters {
	ivec2 size;		// Width, Height.
	vec2 dAdB;		// diffuse A and diffuse B.
	vec2 fk;		// feed rate and kill rate.
};

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

int pos() {
	vec2 p = fragTexCoord * size;
	return (size.x * int(p.y)) + int(p.x);
}

void main() {
	outColor = vec4(hsv2rgb(vec3(AB[pos()].x, 1, 1)), 1.0);
}
