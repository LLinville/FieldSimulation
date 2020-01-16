import math
import random
from PIL import Image
import imageio
from pathlib import Path


import numpy as np
import moderngl as mg
import moderngl_window as mglw
from util import add_point, add_packet, total_mag

GROUP_SIZE = 1024
timestep = 0.005

"""wavefunc_compute_shader_code = '''
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

# with open("rk_grad_compute_shader.glsl") as shader_file:
#     grad_compute_shader_code = shader_file.read()
#
with open("pot_compute_shader.glsl") as shader_file:
    pot_compute_shader_code = shader_file.read()

with open("wavefunc_compute_shader.glsl") as shader_file:
    wavefunc_compute_shader_code = shader_file.read()

with open("rk_grad_compute_shader.glsl") as shader_file:
    rk_grad_compute_shader_code = shader_file.read()

# grad_compute_shader_code = """
# #version 430
# #define GROUP_SIZE """ + str(GROUP_SIZE) + grad_compute_shader_code
#
pot_compute_shader_code = """
#version 430
#define GROUP_SIZE """ + str(GROUP_SIZE) + pot_compute_shader_code

wavefunc_compute_shader_code = """
#version 430
#define TIMESTEP """ + str(timestep) + """
#define GROUP_SIZE """ + str(GROUP_SIZE) + wavefunc_compute_shader_code

rk_grad_compute_shader_code = """
#version 430
#define TIMESTEP """ + str(timestep) + """
#define GROUP_SIZE """ + str(GROUP_SIZE) + rk_grad_compute_shader_code

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

# with open("domain_color_util_1_glsl.txt") as file:
#     util_shader_code = file.read()

with open("adaptive_domain_coloring.glsl") as file:
    util_shader_code = file.read()

# with open("domain_color_2.glsl") as file:
#     util_shader_code = file.read()

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
                        //float mag = sqrt(f_color.x*f_color.x + f_color.y*f_color.y) / 10;
                        //f_color = vec4(mag, mag, mag,0);
                        /*
                        f_color = vec4(domainColoring (
                          f_color.rg * 10, // z
                          vec2(5,5), // vec2 polarGridSpacing
                          0.0, // float polarGridStrength
                          vec2(5,5), // vec2 rectGridSpacing
                          0.1, // float rectGridStrength
                          0.45, // float poleLightening
                          11.8, // float poleLighteningSharpness
                          0.8, // float rootDarkening
                          0.1*1.6, // float rootDarkeningSharpness
                          0.0 // float lineWidth
                        ),1);
                        */
                        
                        f_color = vec4(imagineColor(f_color.xy * 100));
                        
                        //f_color = vec4(complexToColor(f_color.xy * 1));
                    }
                '''
)

texture_depth = 4
vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
# vertices = np.array([0, 0, 0, 1.0, 1.0, 0, 1.0, 1.0])
texture = ctx.texture((width, height), texture_depth, np.zeros((width, height,texture_depth), dtype='f4').tobytes(), dtype='f4')

vbo = ctx.buffer(vertices.astype('f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

initial_data_a = np.zeros((width, height, 4))
initial_data_b = np.zeros_like(initial_data_a)
initial_data_zero = np.zeros_like(initial_data_a)
initial_data_pot = np.zeros((width, height, 2))
# initial_data_pot[:,:,0] = 1
# initial_data_pot[400:600,400:600] = 0
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
# add_point(initial_data_a, 400, 400, turns=1)
# add_point(initial_data_a, 600, 600, turns=1)
# add_packet(initial_data_a, 300, 300, width=20, momentum=15)
add_packet(initial_data_a, 600, 500, width=20, momentum=8, direction=math.pi/2)
context = mg.create_context()
point_buffer_a = context.buffer(np.array(initial_data_a, dtype='f4').tobytes())
point_buffer_b = context.buffer(np.array(initial_data_b, dtype='f4').tobytes())
pot_buffer = context.buffer(np.array(initial_data_pot, dtype='f4').tobytes())
k1_buffer = context.buffer(np.array(initial_data_zero, dtype='f4').tobytes())
k2_buffer = context.buffer(np.array(initial_data_zero, dtype='f4').tobytes())
k3_buffer = context.buffer(np.array(initial_data_zero, dtype='f4').tobytes())
k4_buffer = context.buffer(np.array(initial_data_zero, dtype='f4').tobytes())
rk_step_buffer_a = context.buffer(b'\x01')
rk_step_buffer_b = context.buffer(b'\x01')

starting_mag = total_mag(point_buffer_a)
# grad_buffer = context.buffer(np.array(initial_data_acc, dtype='f4').tobytes())
# grav_buffer = context.buffer(np.array(initial_data_acc, dtype='f4').tobytes())

wavefunc_compute_shader = context.compute_shader(wavefunc_compute_shader_code)
pot_compute_shader = context.compute_shader(pot_compute_shader_code)
rk_grad_compute_shader = context.compute_shader(rk_grad_compute_shader_code)
# grad_compute_shader = context.compute_shader(grad_compute_shader_code)

pot_buffer.bind_to_storage_buffer(3)
k1_buffer.bind_to_storage_buffer(4)
k2_buffer.bind_to_storage_buffer(5)
k3_buffer.bind_to_storage_buffer(6)
k4_buffer.bind_to_storage_buffer(7)
rk_step_buffer_a.bind_to_storage_buffer(8)
rk_step_buffer_b.bind_to_storage_buffer(9)
# grad_buffer.bind_to_storage_buffer(4)
# grav_buffer.bind_to_storage_buffer(5)
# text = ctx.buffer(grid.tobytes())
# tao = ctx.simple_vertex_array(prog, text, 'in_text')


def mouse_func(window, x,y):
    # window.title = f'{x} {y}'
    value = np.reshape(np.frombuffer(point_buffer_a.read(), dtype="f4"), (-1, 1024, 4))[x, 1024-y, 0:2]
    pot = np.reshape(np.frombuffer(pot_buffer.read(), dtype="f4"), (-1, 1024, 2))[x, 1024-y, 0:2]
    print(f'pot: {pot}', value, np.sqrt(np.sum(value * value)))
window.mouse_position_event_func = lambda x,y: mouse_func(window,x,y)

OUTPUT_DIRPATH = "./output3"
Path(OUTPUT_DIRPATH).mkdir(parents=True, exist_ok=True)
imgs = []
toggle = False
for iter in range(2000):
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

        # grad_compute_shader.run(group_x=GROUP_SIZE)
        pot_compute_shader.run(group_x=GROUP_SIZE)

        # set k1, k2, k3, k4

        rk_step_buffer_a.bind_to_storage_buffer(8)
        rk_step_buffer_b.bind_to_storage_buffer(9)
        # print(f'1 in:{rk_step_buffer_a.read()}, out {rk_step_buffer_b.read()}')
        rk_grad_compute_shader.run(group_x=GROUP_SIZE)
        # print(f'2 in:{rk_step_buffer_b.read()}, out {rk_step_buffer_a.read()}')
        rk_step_buffer_b.bind_to_storage_buffer(8)
        rk_step_buffer_a.bind_to_storage_buffer(9)
        # print(f'2 in:{rk_step_buffer_b.read()}, out {rk_step_buffer_a.read()}')
        rk_grad_compute_shader.run(group_x=GROUP_SIZE)
        # print(f'3 in:{rk_step_buffer_a.read()}, out {rk_step_buffer_b.read()}')
        rk_step_buffer_a.bind_to_storage_buffer(8)
        rk_step_buffer_b.bind_to_storage_buffer(9)
        # print(f'3 in:{rk_step_buffer_a.read()}, out {rk_step_buffer_b.read()}')
        rk_grad_compute_shader.run(group_x=GROUP_SIZE)
        # print(f'4 in:{rk_step_buffer_b.read()}, out {rk_step_buffer_a.read()}')
        rk_step_buffer_b.bind_to_storage_buffer(8)
        rk_step_buffer_a.bind_to_storage_buffer(9)
        # print(f'4 in:{rk_step_buffer_b.read()}, out {rk_step_buffer_a.read()}')
        rk_grad_compute_shader.run(group_x=GROUP_SIZE)
        # print(f'1 in:{rk_step_buffer_a.read()}, out {rk_step_buffer_b.read()}')

        wavefunc_compute_shader.run(group_x=GROUP_SIZE)

    # print(iter)
    window.clear()
    ctx.clear(1.0, 1.0, 1.0)
    # texture.write(acc_buffer.read())
    # texture.write(grad_buffer.read())
    points = np.reshape(np.frombuffer(point_buffer_a.read(), dtype="f4"), (-1, 1024, 4))
    total = total_mag(point_buffer_a)
    # print(total)
    # starting_mag = 1
    # print(f'total mag: {total} \t normalized ratio: {starting_mag / total}')
    if not 100 < starting_mag / total < 0.01:
        # print("RENORMALIZING")
        normalized = points * starting_mag / total
        point_buffer_a.write(normalized.tobytes())
        texture.write(normalized.tobytes())
    else:
        texture.write(point_buffer_a)
    # grav_data = np.frombuffer(grav_buffer.read(), dtype="f4")
    # texture.write(grav_buffer.read())

    if iter % 10 == 0:
        print(iter)



    texture.use()
    vao.render(mg.TRIANGLE_STRIP)
    window.swap_buffers()
    # output = np.frombuffer(window.fbo.read(), dtype=np.int)
    # output = output.reshape((1024, 1024, 4))
    # output = np.multiply(output, 255).astype(np.uint8)
    # imageio.imwrite(f'{OUTPUT_DIRPATH}/{iter}.png', output)
    # imgs.append(output)
    output = Image.frombuffer('RGB', window.fbo.size, window.fbo.read(), 'raw', 'RGB', 0, -1)
    output.save(f'{OUTPUT_DIRPATH}/{iter}.png')


# out_path = f"{OUTPUT_DIRPATH}/debug.gif"
# print("Writing GIF anim to", out_path)
# imageio.mimwrite(out_path, imgs, "GIF")

    # print(f'total_grav {np.sum(np.frombuffer(grav_buffer.read(), dtype="f4"))}')


# config = mglw.WindowConfig(context)
# config.window_size = window_sizeS
# config.aspect_ratio = aspect_ratio
# config.gl_version = gl_version
#
# mglw.run_window_config(config)