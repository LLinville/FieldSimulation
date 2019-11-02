import numpy as np
import cmath
from math import sqrt

import moderngl
from window import Example

def invert(x, y):
    mag = x*x+y*y
    return min(x / mag, 1), min(y / mag, 1)


def add_point(field, location, size=10, polarity=1, rotation = 0, turns=1):
    patch_input_x, patch_input_y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    patch = np.zeros((size, size), dtype=complex)
    for u in range(size):
        for v in range(size):
            input_point = patch_input_x[u,v] + 1j * patch_input_y[u,v]
            r, theta = np.abs(input_point), np.angle(input_point)
            theta *= turns
            r = 1 / np.abs(r * 10)
            input_point = cmath.rect(r, theta)
            dist2 = (u) ** 2 + (v) ** 2 + 1
            patch[u,v] += input_point

    rotated = patch * (np.cos(rotation) + np.sin(rotation) * 1j)

    field[location[0] - size // 2 : location[0] + size // 2, location[1] - size // 2 : location[1] + size // 2, 0] += rotated.real
    field[location[0] - size // 2: location[0] + size // 2, location[1] - size // 2: location[1] + size // 2, 1] += rotated.imag


class ComplexIntegrator(Example):
    title = "Polar field"
    aspect_ratio = 1.0
    window_size = (500, 500)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        width, height = self.window_size
        canvas = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).astype('f4')
        pixels = np.random.rand(width, height,2).astype('f4')
        pixels = np.array([
            [[invert(x, y)] for y in np.linspace(-10, 10, width)] for x in np.linspace(-10, 10, height)
        ]).astype('f4')
        pixels = np.zeros((width, height, 2)).astype('f4')
        # add_point(pixels, (200, 200), 50)
        add_point(pixels, (100, 100), 100, turns=1)
        grid = np.dstack(np.mgrid[0:height, 0:width][::-1]).astype('i4')

        starting_positions = np.zeros_like(pixels).astype('f4')
        starting_positions = pixels.copy()
        starting_velocities = np.zeros_like(pixels).astype('f4')

        self.step_positions = self.ctx.program(
            vertex_shader='''
                        #version 330

                        uniform sampler2D VelocitiesTexture;
                        uniform sampler2D PositionsTexture;
                        uniform int Width;
                        uniform int Height;

                        in ivec2 in_coords;
                        out vec2 out_pos;
                        out vec2 out_acc;
                        
                        const float NeighborForceMagnitude = 1.0;
                        const float AbsoluteForceMagnitude = -1.0;
                        const float Timestep = 0.1;

                        vec2 vel(int x, int y) {
                            return texelFetch(VelocitiesTexture, ivec2((x + Width) % Width, (y + Height) % Height), 0).rg;
                        }
                        
                        vec2 pos(int x, int y) {
                            return texelFetch(PositionsTexture, ivec2((x + Width) % Width, (y + Height) % Height), 0).rg;
                        }

                        void main() {
                            vec2 neighborAverage = vec2(0.0, 0.0);

                            neighborAverage += pos(in_coords.x - 1, in_coords.y);
                            neighborAverage += pos(in_coords.x + 1, in_coords.y);
                            neighborAverage += pos(in_coords.x, in_coords.y - 1);   
                            neighborAverage += pos(in_coords.x, in_coords.y + 1);

                            neighborAverage /= 4.0;
                            
                            out_pos = pos(in_coords.x, in_coords.y);
                            out_acc = NeighborForceMagnitude * (neighborAverage - out_pos) + AbsoluteForceMagnitude * out_pos;
                            
                            out_pos *= (1.0 - Timestep);
                            out_pos += Timestep * (vel(in_coords.x, in_coords.y) + Timestep * out_acc / 2.0);
                        }
                    ''',
            varyings=['out_pos']
        )

        self.step_velocities = self.ctx.program(
            vertex_shader='''
                                #version 330

                                uniform sampler2D VelocitiesTexture;
                                uniform sampler2D PositionsTexture;
                                uniform sampler2D AccelerationsTexture;
                                uniform int Width;
                                uniform int Height;

                                in ivec2 in_coords;
                                out vec2 out_vel;

                                const float ForceMagnitude = -1.0;
                                const float Timestep = 0.01;

                                vec2 vel(int x, int y) {
                                    return texelFetch(VelocitiesTexture, ivec2((x + Width) % Width, (y + Height) % Height), 0).rg;
                                }

                                vec2 pos(int x, int y) {
                                    return texelFetch(PositionsTexture, ivec2((x + Width) % Width, (y + Height) % Height), 0).rg;
                                }
                                
                                vec2 acc(int x, int y) {
                                    return texelFetch(AccelerationsTexture, ivec2((x + Width) % Width, (y + Height) % Height), 0).rg;
                                }

                                void main() {
                                    vec2 neighborAverage = vec2(0.0, 0.0);

                                    neighborAverage += pos(in_coords.x - 1, in_coords.y);
                                    neighborAverage += pos(in_coords.x + 1, in_coords.y);
                                    neighborAverage += pos(in_coords.x, in_coords.y - 1);
                                    neighborAverage += pos(in_coords.x, in_coords.y + 1);

                                    neighborAverage = neighborAverage / 4.0;

                                    vec2 force = ForceMagnitude * (neighborAverage - pos(in_coords.x, in_coords.y));

                                    out_vel = vel(in_coords.x, in_coords.y);
                                    out_vel += Timestep * (acc(in_coords.x, in_coords.y) + force) / 2.0;
                                }
                            ''',
            varyings=['out_vel']
        )

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                in vec2 in_vert;
                out vec2 v_text;
                
                void main() {
                    v_text = in_vert;
                    gl_Position = vec4(in_vert * 2.0 - 1.0, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                
                const float PI = 3.1415926535897932384626433832795;

                uniform sampler2D Texture;

                in vec2 v_text;
                out vec4 f_color;
                
                vec3 hsv2rgb(vec3 c) {
                  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec4 input = texture(Texture, v_text);
                    float mag = input.r*input.r+input.g*input.g;
                    mag = sqrt(mag);
                    vec3 hsv = vec3(atan(input.r, input.g)/(2*PI), 1, mag * 0.7 + 0.3);
                    vec3 rgb = hsv2rgb(hsv);
                    f_color = vec4(rgb.rgb, 1);
                }
            ''',
        )

        self.step_positions['Width'].value = width
        self.step_positions['Height'].value = height

        self.step_velocities['Width'].value = width
        self.step_velocities['Height'].value = height

        self.positions = self.ctx.texture((width, height), 2, starting_positions.tobytes(), dtype='f4')
        self.positions.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.positions.swizzle = 'RGG1'
        self.positions.use()

        self.velocities = self.ctx.texture((width, height), 2, starting_velocities.tobytes(), dtype='f4')
        self.velocities.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.velocities.swizzle = 'RGG1'
        self.velocities.use()

        self.vbo = self.ctx.buffer(canvas.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

        self.text = self.ctx.buffer(grid.tobytes())
        self.pos_tao = self.ctx.simple_vertex_array(self.step_positions, self.text, 'in_coords')
        self.pos_pbo = self.ctx.buffer(reserve=pixels.nbytes)

        self.vel_tao = self.ctx.simple_vertex_array(self.step_velocities, self.text, 'in_coords')
        self.vel_pbo = self.ctx.buffer(reserve=pixels.nbytes)

    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)

        self.pos_tao.transform(self.pos_pbo)
        self.positions.write(self.pos_pbo)

        self.vel_tao.transform(self.vel_pbo)
        self.velocities.write(self.vel_pbo)

        self.vao.render(moderngl.TRIANGLE_STRIP)


if __name__ == '__main__':
    ComplexIntegrator.run()
