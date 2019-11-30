import math
import random

import numpy as np
import moderngl as mg
import moderngl_window as mglw

GROUP_SIZE = 1024

"""compute_shader_code = '''
version 330

const float PI = 3.1415926535897932384626433832795;

uniform sampler2D Texture;

in vec2 v_text;
out vec4 f_color;

const int Width = 500;
const int Height = 500;

vec2 lookup(int x, int y) {
    return texture(Texture, vec2((x + Width) % Width, (y + Height) % Height)).rg;
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {

    //vec4 input = texture(Texture, vec2((v_text.x - 1 + Width) % Width, (v_text.y + Height) % Height));
    //vec2 input = texture(Texture, vec2(v_text.r - 1, v_text.g)).rg;
    // vec2 input = texelFetch(Texture, v_text).rg;
    vec2 input = texture(Texture, vec2(v_text.r, v_text.g)).rg;
    
    vec2 left = texture(Texture, vec2(v_text.r - 1, v_text.g)).rg;
    vec2 right = texture(Texture, vec2(v_text.r + 1, v_text.g)).rg;
    vec2 down = texture(Texture, vec2(v_text.r, v_text.g - 1)).rg;
    vec2 up = texture(Texture, vec2(v_text.r, v_text.g + 1)).rg;
    //*/
    // out_vert = (left + right + up + down + center) / 5.0;
    vec2 grad = vec2(right - left, up - down) / 2.0;
    if (grad.r*grad.r+grad.g*grad.g < 0.00000000000) {
        input = grad;
    }
    
    float mag = input.r*input.r+input.g*input.g;
    mag = sqrt(mag);
    vec3 hsv = vec3(atan(input.r, input.g)/(2*PI), 1, mag);// * 0.7 + 0.3);
    vec3 rgb = hsv2rgb(hsv);
    f_color = vec4(rgb.rgb, 1);
}
'''"""
frag_shader_code = """
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
"""
compute_shader_code = """
#version 430
#define GROUP_SIZE """ + str(GROUP_SIZE) + """

layout(local_size_x=GROUP_SIZE) in;

struct FieldPoint
{
    vec2 pos;
    vec2 vel;
};

layout(std430, binding=1) buffer field_in
{
    FieldPoint fieldPoints[];
} In;
layout(std430, binding=2) buffer field_out
{
    FieldPoint fieldPoints[];
} Out;

FieldPoint getAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return In.fieldPoints[(y) * width + x];
}

void main()
{
    int width = 1024;
    int height = 1024;
    int n = int(gl_GlobalInvocationID);
    int x = n % width;
    int y = n / width;
    
    FieldPoint in_point = getAt(x,y);

    vec2 neighbor_pos_diff = -4.0 * in_point.pos
        + getAt(x-1, y).pos
        + getAt(x+1, y).pos
        + getAt(x, y-1).pos
        + getAt(x, y+1).pos;
    
    neighbor_pos_diff = neighbor_pos_diff / 4;
    
    vec2 out_pos = in_point.pos + in_point.vel * 0.00001;
    vec2 out_vel = in_point.vel - neighbor_pos_diff * 0.01;

    FieldPoint out_point;
    out_point.pos.xy = out_pos;
    out_point.vel.xy = out_vel;
    //In.fieldPoints[n].pos = vec2(1.0,1.0);
    //Out.fieldPoints[n].pos = vec2(0.4,0.6);
    Out.fieldPoints[n] = out_point;
}
"""
gl_version = (4, 3)
title = "ModernGL shader view"
window_size = (1024, 1024)
height, width = window_size
aspect_ratio = 1 / 1
resizable = True
samples = 4
window_str = 'moderngl_window.context.pyglet.Window'
window_cls = mglw.get_window_cls(window_str)
window = window_cls(
    title="My Window",
    gl_version=(4, 1),
    size=(1024, 1024),
    aspect_ratio=aspect_ratio,

)
ctx = window.ctx
mglw.activate_context(ctx=ctx)

prog = ctx.program(
            vertex_shader='''
                #version 430

                in vec2 in_vert;
                out vec2 v_text;

                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_text = in_vert;
                }
            ''',
            fragment_shader='''
                #version 430

                in vec2 v_text;
                out vec4 f_color;

                uniform sampler2D Texture;

                void main() {
                    f_color = texture(Texture, vec2(v_text.x / 2.0 - 1.5, v_text.y / 2.0 - 1.5));
                    //f_color = vec4(0, 0, f_color.b, 0);
                }
            '''
        )

vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
# vertices = np.array([0, 0, 0, 1.0, 1.0, 0, 1.0, 1.0])
texture = ctx.texture((width, height), 4, np.zeros((width, height, 4), dtype='f4').tobytes(), dtype='f4')

vbo = ctx.buffer(vertices.astype('f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

initial_data_a = np.zeros((width, height, 4))
initial_data_b = np.zeros((width, height, 4))
# initial_data_a[150:-150,150:-150,0] = 1.0
# initial_data_a[150:-150,150:-150,2] = 0.0
for x in range(width):
    for y in range(height):
        center = (width // 2, height // 2)
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        if (dist < 100):
            initial_data_a[x,y] = (100 - dist) /100

context = mg.create_context()
compute_buffer_a = context.buffer(np.array(initial_data_a, dtype='f4').tobytes())
compute_buffer_b = context.buffer(np.array(initial_data_b, dtype='f4').tobytes())

compute_shader = context.compute_shader(compute_shader_code)
# text = ctx.buffer(grid.tobytes())
# tao = ctx.simple_vertex_array(prog, text, 'in_text')

toggle = False
for iter in range(100000):
    for substep in range(100):

        toggle = not toggle
        if toggle:
            a, b = 1, 2
            target_buffer = compute_buffer_b
        else:
            a, b = 2, 1
            target_buffer = compute_buffer_a



        compute_buffer_a.bind_to_storage_buffer(a)
        compute_buffer_b.bind_to_storage_buffer(b)
        compute_shader.run(group_x=GROUP_SIZE)

    print(iter)
    window.clear()
    ctx.clear(1.0, 1.0, 1.0)
    texture.write(target_buffer.read())

    texture.use()
    vao.render(mg.TRIANGLE_STRIP)
    window.swap_buffers()








# config = mglw.WindowConfig(context)
# config.window_size = window_size
# config.aspect_ratio = aspect_ratio
# config.gl_version = gl_version
#
# mglw.run_window_config(config)