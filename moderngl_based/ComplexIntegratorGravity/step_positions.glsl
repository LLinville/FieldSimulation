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
const float Timestep = 0.01;

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