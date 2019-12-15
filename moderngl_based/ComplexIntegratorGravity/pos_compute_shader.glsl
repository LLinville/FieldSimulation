
#define TIMESTEP 0.001

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

    float ACC_STRENGTH = 10;

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


    vec2 out_pos = in_point.pos + TIMESTEP * (in_point.vel + neighbor_pos_diff*0 + ACC_STRENGTH*TIMESTEP * acc / 2.0);
    vec2 out_vel = in_point.vel + ACC_STRENGTH * TIMESTEP * (acc + neighbor_pos_diff ) / 2.0;
    //out_vel = neighbor_pos_diff * 10;

    //vec2 out_pos = in_point.pos + neighbor_pos_diff * TIMESTEP * 10000;
    //vec2 out_vel = in_point.vel + neighbor_pos_diff * 1.0;
    //out_vel *= 0.99999;

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