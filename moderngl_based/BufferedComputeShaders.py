
'''
    example of using compute shader.

    requirements:
     - numpy
     - imageio (for output)
'''

import os

import moderngl
import numpy as np
import imageio  # for output

velocity_step_shader_code = '''
    #version 330
    
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
            '''

def source(uri, consts):
    ''' read gl code '''
    with open(uri, 'r') as fp:
        content = fp.read()

    # feed constant values
    for key, value in consts.items():
        content = content.replace(f"%%{key}%%", str(value))
    return content


# W = X * Y  // for each run, handles a row of pixels
# execute compute shader for H times to complete
W = 512
H = 256
X = W
Y = 1
Z = 1
consts = {
    "W": W,
    "H": H,
    "X": X + 1,
    "Y": Y,
    "Z": Z,
}

FRAMES = 90
OUTPUT_DIRPATH = "./output"

if not os.path.isdir(OUTPUT_DIRPATH):
    os.makedirs(OUTPUT_DIRPATH)

context = moderngl.create_standalone_context(require=430)
compute_shader = context.compute_shader(source('./gl/median_5x5.gl', consts))

# init buffers
buffer_a_data = np.random.uniform(0.0, 1.0, (H, W, 4)).astype('f4')
buffer_a = context.buffer(buffer_a_data)
buffer_b_data = np.zeros((H, W, 4)).astype('f4')
buffer_b = context.buffer(buffer_b_data)

imgs = []
last_buffer = buffer_b
for i in range(FRAMES):
    toggle = True if i % 2 else False
    buffer_a.bind_to_storage_buffer(1 if toggle else 0)
    buffer_b.bind_to_storage_buffer(0 if toggle else 1)

    # toggle 2 buffers as input and output
    last_buffer = buffer_a if toggle else buffer_b

    # local invocation id x -> pixel x
    # work groupid x -> pixel y
    # eg) buffer[x, y] = gl_LocalInvocationID.x + gl_WorkGroupID.x * W
    compute_shader.run(group_x=H, group_y=1)

    # print out
    output = np.frombuffer(last_buffer.read(), dtype=np.float32)
    output = output.reshape((H, W, 4))
    output = np.multiply(output, 255).astype(np.uint8)
    imgs.append(output)

# if you don't want to use imageio, remove this section
out_path = f"{OUTPUT_DIRPATH}/debug.gif"
print("Writing GIF anim to", out_path)
imageio.mimwrite(out_path, imgs, "GIF", duration=0.15)
