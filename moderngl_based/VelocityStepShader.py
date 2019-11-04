from moderngl_based import Program
import numpy as np


class VelocityStepShader(Program):


    def __init__(self, ctx):
        super().__init__(ctx)
        width, height = (1000, 1000)
        self.program = self.ctx.(
            vertex_shader='''
                #version 330

                uniform vec2 Velocities;
                uniform vec2 Positions;
                uniform vec2 Accelerations;
                uniform int Width;
                uniform int Height;

                in ivec2 in_coords;
                out vec2 out_vel;

                const float ForceMagnitude = -1.0;
                const float Timestep = 0.01;

                vec2 vel(int x, int y) {
                    return Velocities[(x + Width) % Width][(y + Height) % Height].rg;
                }

                vec2 pos(int x, int y) {
                    return Positions[(x + Width) % Width][(y + Height) % Height].rg;
                }

                vec2 acc(int x, int y) {
                    return Accelerations[(x + Width) % Width][(y + Height) % Height].rg;
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
        # vtx -> tesselate -> geom -> fragment

        pixels = np.zeros((width, height, 2)).astype('f4')
        starting_velocities = np.zeros_like(pixels).astype('f4')
        self.velocities = self.ctx.vertex_array((width, height), 2, starting_velocities.tobytes(), dtype='f4')
        self.step_velocities['Width'].value = width
        self.step_velocities['Height'].value = height
        self.step_velocities['PositionsTexture'] = self.positions
        self.step_velocities['VelocitiesTexture'] = self.velocities

        # set up some parameters in the shader
        self.program['Velocities'] .value = makeIdentityMatrix(4)
        self.program['enableTexture'].value = False
        self.program['minAlpha']     .value = 0.5
        self.program['modColor']     .value = (1.0, 1.0, 1.0, 1.0)

        # make vertex buffer/attribute objects to hold the line data
        self.pointDataSize = struct.calcsize(self.POINT_FMT)
        self.lineDataSize  = struct.calcsize(self.LINE_FMT)

        self.vboVtxs = self.ctx.buffer( # the actual vertices
            reserve=self.MAX_POINTS * self.pointDataSize,
            dynamic=True,
        )
        self.iboLines = self.ctx.buffer( # vtx index buffer
            reserve=self.MAX_POINTS * self.lineDataSize,
            dynamic=True,
        )

        self.vao = self.ctx.vertex_array(self.program,
            [ # inputs to the vertex shader
                (self.vboVtxs, '3f', 'in_p0'),
                (self.vboVtxs, '3f', 'in_p1'),
                (self.vboVtxs, '3f', 'in_c0'),
                (self.vboVtxs, '3f', 'in_c1'),
            ],
            #self.iboLines,
            None,
        )
        #self.vboVtxs.bind_to_uniform_block(
        #    self.program['vertices'].location)


    def setVertices(self, idx, *points):
        """Change one or more vertices in the buffer.
        idx:    Point index to set in the buffer.
        points: Point to write.
        """
        if idx < 0: idx = self.MAX_POINTS + idx
        if idx < 0 or idx + len(points) >= self.MAX_POINTS:
            raise IndexError(idx)

        data = []
        for p in points:
            data.append(struct.pack(self.POINT_FMT, *p))

        self.vboVtxs.write(b''.join(data),
            offset = idx * self.pointDataSize)


    def setLines(self, idx, *lines):
        """Change one or more lines in the buffer.
        idx:   Line to set.
        lines: Vertex indices to write. (p0, p1, c0, c1)
        """
        # XXX should we be using MAX_POINTS here?
        if idx < 0: idx = self.MAX_POINTS + idx
        if idx < 0 or idx + len(lines) >= self.MAX_POINTS:
            raise IndexError(idx)

        data = []
        for line in lines:
            p0, p1, c0, c1 = line
            if c0 is None: c0 = p0
            if c1 is None: c1 = c0
            #p0 = 3
            #p1 = 1
            #c0 = 2
            #c1 = 0
            data.append(struct.pack(self.LINE_FMT, p0, p1, c0, c1))

        self.iboLines.write(b''.join(data),
            offset = idx * self.lineDataSize)


    def run(self):
        """Draw the lines."""

        #data = self.iboLines.read()
        #dump = []
        #for i in range(0, 0x100, 16):
        #    line = "%04X " % i
        #    for j in range(16):
        #        if (j&3) == 0: line += ' '
        #        line += "%02X " % data[i+j]
        #    dump.append(line)
        #print("index buffer (obj %d):\n%s" % (
        #    self.iboLines.glo,
        #    '\n'.join(dump),
        #))
        checkError(self, "run 1")

        # update projection matrix to current viewport
        x, y, width, height = self.ctx.viewport
        self.program['matProjection'].value = \
            makePerspectiveMatrix(0, width, height, 0, 1, 100)
        checkError(self, "run 2")

        p0  = self.program['in_p0'].location
        p1  = self.program['in_p1'].location
        c0  = self.program['in_c0'].location
        c1  = self.program['in_c1'].location
        vbo = self.vboVtxs
        checkError(self, "run 3")
        print("p0=", p0, "p1=", p1, "c0=", c0, "c1=", c1)

        self.vao.bind(p0, 'f', vbo, '3f', offset=0, stride=12*4)
        checkError(self, "bind 1")
        self.vao.bind(p1, 'f', vbo, '3f', offset=12)
        self.vao.bind(c0, 'f', vbo, '3f', offset=24)
        self.vao.bind(c1, 'f', vbo, '3f', offset=36)
        checkError(self, "run 4")
        self.vao.render(mode=moderngl.POINTS, vertices=4, instances=4)
        checkError(self, "run 5")