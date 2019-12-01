import math
import random


import numpy as np
import moderngl as mg
import moderngl_window as mglw
from util import add_point

GROUP_SIZE = 1024

"""pos_compute_shader_code = '''
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
grad_compute_shader_code = """
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
layout(std430, binding=3) buffer acc_in
{
    vec2 acc[];
} Acc;
layout(std430, binding=4) buffer grad_in
{
    vec2 grad[];
} Grad;

FieldPoint pointAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return In.fieldPoints[(y) * width + x];
}

vec2 accAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return Acc.acc[(y) * width + x];
}

float mag(vec2 vin){
    return sqrt(dot(vin,vin));
}

void main()
{
    int width = 1024;
    int height = 1024;
    int n = int(gl_GlobalInvocationID);
    int x = n % width;
    int y = n / width;

    vec2 grad = vec2(
        (mag(pointAt(x+1, y).pos) - mag(pointAt(x-1, y).pos)) / 2,
        (mag(pointAt(x, y+1).pos) - mag(pointAt(x, y-1).pos)) / 2
    );
    Grad.grad[n] = grad;
}

"""
acc_compute_shader_code = """
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
layout(std430, binding=3) buffer acc_in
{
    vec2 acc[];
} Acc;

FieldPoint pointAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return In.fieldPoints[(y) * width + x];
}

vec2 accAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return Acc.acc[(y) * width + x];
}

void main()
{
    int width = 1024;
    int height = 1024;
    int n = int(gl_GlobalInvocationID);
    int x = n % width;
    int y = n / width;
    
    FieldPoint in_point = pointAt(x,y);

    vec2 neighbor_pos_diff = -4.0 * in_point.pos
        + pointAt(x-1, y).pos
        + pointAt(x+1, y).pos
        + pointAt(x, y-1).pos
        + pointAt(x, y+1).pos;
    
    Acc.acc[n] = 0*neighbor_pos_diff / 4.0;
}

"""
pos_compute_shader_code = """
#version 430
#define GROUP_SIZE """ + str(GROUP_SIZE) + """
#define TIMESTEP 0.00001

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
layout(std430, binding=3) buffer acc_in
{
    vec2 acc[];
} Acc;

layout(std430, binding=4) buffer grad_in
{
    vec2 grad[];
} Grad;

FieldPoint pointAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return In.fieldPoints[(y) * width + x];
}

vec2 accAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return Acc.acc[(y) * width + x];
}

vec2 gradAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return Grad.grad[(y) * width + x];
}

float exchanged(int x, int y, int dx, int dy) {
    vec2 avg_grad = (gradAt(x,y) + gradAt(x+dx, y+dy)) / 2;
    
    return avg_grad.x * dx + avg_grad.y * dy;
}

void main()
{
    int width = 1024;
    int height = 1024;
    int n = int(gl_GlobalInvocationID);
    int x = n % width;
    int y = n / width;
    
    float ACC_STRENGTH = 1;
    
    FieldPoint in_point = pointAt(x,y);
    
    vec2 acc = accAt(x,y);
    
    vec2 neighbor_pos_diff = -4.0 * in_point.pos
        + pointAt(x-1, y).pos
        + pointAt(x+1, y).pos
        + pointAt(x, y-1).pos
        + pointAt(x, y+1).pos;
        
    neighbor_pos_diff /= 4.0;
    //neighbor_pos_diff *= 0.0;
    
    vec2 neighbor_avg = pointAt(x-1, y).pos
        + pointAt(x+1, y).pos
        + pointAt(x, y-1).pos
        + pointAt(x, y+1).pos;
    neighbor_avg /= 4.0;
    
    
    //vec2 out_pos = in_point.pos + TIMESTEP * (in_point.vel + ACC_STRENGTH*TIMESTEP * acc / 2.0);
    //vec2 out_vel = in_point.vel + TIMESTEP * (ACC_STRENGTH * acc + neighbor_pos_diff ) / 2.0;
    
    vec2 out_pos = in_point.pos + neighbor_pos_diff * TIMESTEP * 10000;
    vec2 out_vel = in_point.vel + neighbor_pos_diff * 1.0;
    out_vel *= 0;
    
    float mag_to_give = exchanged(x,y, 0, 1)
        + exchanged(x,y, 0, -1)
        + exchanged(x,y, 1, 0)
        + exchanged(x,y, -1, 0);
    
    float pos_mag = sqrt(dot(out_pos, out_pos));
    if(mag_to_give > 0.000000){
        //out_pos *= 1.1;
    }

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

with open("adaptive_domain_coloring_glsl.txt") as file:
    util_shader_code = file.read()

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
                
                '''
                + util_shader_code +
                '''

                void main() {
                    f_color = texture(Texture, vec2(v_text.x / 2.0 - 1.5, v_text.y / 2.0 - 1.5));
                    //float mag = sqrt(f_color.x*f_color.x + f_color.y*f_color.y);
                    //f_color = vec4(mag, mag, mag, 1);
                    f_color = vec4(imagineColor(f_color.xy * 100));
                }
            '''
        )

vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
# vertices = np.array([0, 0, 0, 1.0, 1.0, 0, 1.0, 1.0])
texture = ctx.texture((width, height), 4, np.zeros((width, height, 4), dtype='f4').tobytes(), dtype='f4')

vbo = ctx.buffer(vertices.astype('f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

initial_data_a = np.zeros((width, height, 4))
initial_data_b = np.zeros_like(initial_data_a)
initial_data_acc = np.zeros_like(initial_data_a)
# initial_data_a[150:-150,150:-150,0] = 1.0
# initial_data_a[150:-150,150:-150,2] = 0.0
# for x in range(width):
#     for y in range(height):
#         center = (width // 2, height // 2)
#         dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
#         if 1<dist < 100:
#             val = complex(x - center[0], y - center[0]) / 100
#             val = val * (100 - dist) / 100
#             initial_data_a[x,y,0] = val.real
#             initial_data_a[x,y,1] = val.imag

# for x in range(width):
#     for y in range(height):
#         center = (width // 2, height // 2)
#         dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
#         if (dist < 100):
#             initial_data_a[x,y,0] = (100 - dist) / 100
add_point(initial_data_a, 400, 400, turns=1)
add_point(initial_data_a, 600, 600, turns=1)

context = mg.create_context()
point_buffer_a = context.buffer(np.array(initial_data_a, dtype='f4').tobytes())
point_buffer_b = context.buffer(np.array(initial_data_b, dtype='f4').tobytes())
acc_buffer = context.buffer(np.array(initial_data_acc, dtype='f4').tobytes())
grad_buffer = context.buffer(np.array(initial_data_acc, dtype='f4').tobytes())

pos_compute_shader = context.compute_shader(pos_compute_shader_code)
acc_compute_shader = context.compute_shader(acc_compute_shader_code)
grad_compute_shader = context.compute_shader(grad_compute_shader_code)
acc_buffer.bind_to_storage_buffer(3)
grad_buffer.bind_to_storage_buffer(4)
# text = ctx.buffer(grid.tobytes())
# tao = ctx.simple_vertex_array(prog, text, 'in_text')

toggle = False
for iter in range(100000):
    for substep in range(1000):

        toggle = not toggle
        if toggle:
            a, b = 1, 2
            target_buffer = point_buffer_b
        else:
            a, b = 2, 1
            target_buffer = point_buffer_a



        point_buffer_a.bind_to_storage_buffer(a)
        point_buffer_b.bind_to_storage_buffer(b)

        grad_compute_shader.run(group_x=GROUP_SIZE)
        acc_compute_shader.run(group_x=GROUP_SIZE)
        pos_compute_shader.run(group_x=GROUP_SIZE)


    print(iter)
    window.clear()
    ctx.clear(1.0, 1.0, 1.0)
    # texture.write(acc_buffer.read())
    texture.write(grad_buffer.read())
    texture.use()
    vao.render(mg.TRIANGLE_STRIP)
    window.swap_buffers()








# config = mglw.WindowConfig(context)
# config.window_size = window_size
# config.aspect_ratio = aspect_ratio
# config.gl_version = gl_version
#
# mglw.run_window_config(config)