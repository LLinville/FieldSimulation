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