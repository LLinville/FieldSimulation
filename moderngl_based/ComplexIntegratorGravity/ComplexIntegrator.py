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
with open("fragment_shader.glsl") as shader_file:
    frag_shader_code = shader_file.read()

with open("grad_compute_shader.glsl") as shader_file:
    grad_compute_shader_code = shader_file.read()

with open("acc_compute_shader.glsl") as shader_file:
    acc_compute_shader_code = shader_file.read()

with open("pos_compute_shader.glsl") as shader_file:
    pos_compute_shader_code = shader_file.read()

grad_compute_shader_code = """
#version 430
#define GROUP_SIZE """ + str(GROUP_SIZE) + grad_compute_shader_code

acc_compute_shader_code = """
#version 430
#define GROUP_SIZE """ + str(GROUP_SIZE) + acc_compute_shader_code

pos_compute_shader_code = """
#version 430
#define GROUP_SIZE """ + str(GROUP_SIZE) + pos_compute_shader_code

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
    size=(1324, 1324),
    aspect_ratio=aspect_ratio,

)
ctx = window.ctx
mglw.activate_context(ctx=ctx)

with open("domain_color_util_1_glsl.txt") as file:
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
                        //f_color = vec4(mag, mag, mag,0);
                        f_color = vec4(domainColoring (
                          f_color.rg * 10, // z
                          vec2(5,5), // vec2 polarGridSpacing
                          0.0, // float polarGridStrength
                          vec2(5,5), // vec2 rectGridSpacing
                          0.1, // float rectGridStrength
                          0.45, // float poleLightening
                          11.8, // float poleLighteningSharpness
                          1.1, // float rootDarkening
                          1*1.6, // float rootDarkeningSharpness
                          0.0 // float lineWidth
                        ),1);
                        //f_color = vec4(imagineColor(f_color.xy * 100));
                    }
                '''
)

vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
# vertices = np.array([0, 0, 0, 1.0, 1.0, 0, 1.0, 1.0])
texture = ctx.texture((width, height), 2, np.zeros((width, height,2), dtype='f4').tobytes(), dtype='f4')

vbo = ctx.buffer(vertices.astype('f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

initial_data_a = np.zeros((width, height, 4))
initial_data_b = np.zeros_like(initial_data_a)
initial_data_acc = np.zeros((width, height, 2))
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
add_point(initial_data_a, 600, 600, turns=0)

context = mg.create_context()
point_buffer_a = context.buffer(np.array(initial_data_a, dtype='f4').tobytes())
point_buffer_b = context.buffer(np.array(initial_data_b, dtype='f4').tobytes())
acc_buffer = context.buffer(np.array(initial_data_acc, dtype='f4').tobytes())
grad_buffer = context.buffer(np.array(initial_data_acc, dtype='f4').tobytes())
grav_buffer = context.buffer(np.array(initial_data_acc, dtype='f4').tobytes())

pos_compute_shader = context.compute_shader(pos_compute_shader_code)
acc_compute_shader = context.compute_shader(acc_compute_shader_code)
grad_compute_shader = context.compute_shader(grad_compute_shader_code)
acc_buffer.bind_to_storage_buffer(3)
grad_buffer.bind_to_storage_buffer(4)
grav_buffer.bind_to_storage_buffer(5)
# text = ctx.buffer(grid.tobytes())
# tao = ctx.simple_vertex_array(prog, text, 'in_text')

toggle = False
for iter in range(100000):
    for substep in range(10):

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
    # texture.write(point_buffer_a.read())
    # texture.write(grav_buffer.read())
    texture.use()
    vao.render(mg.TRIANGLE_STRIP)
    window.swap_buffers()

    print(f'total_grav {np.sum(np.frombuffer(grav_buffer.read(), dtype="f4"))}')


# config = mglw.WindowConfig(context)
# config.window_size = window_size
# config.aspect_ratio = aspect_ratio
# config.gl_version = gl_version
#
# mglw.run_window_config(config)