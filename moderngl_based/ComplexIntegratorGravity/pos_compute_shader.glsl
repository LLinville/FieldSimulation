
#define TIMESTEP 0.01

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

layout(std430, binding=5) buffer grav_in
{
    vec2 grav[];
} Grav;

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

vec2 gravAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return Grav.grav[(y) * width + x];
}

float received(int x, int y, int dx, int dy) {
    vec2 avg_grad = (gradAt(x,y) + gradAt(x+dx, y+dy)) / 2;
    vec2 pos = pointAt(x,y).pos;

    return (sqrt(pos.x*pos.x+pos.y*pos.y))*(avg_grad.x * -1 * dx + avg_grad.y * -1 * dy);
}

vec2 spread_received(int x, int y, int dx, int dy) {
    vec2 pos_center = pointAt(x,y).pos;
    vec2 pos_2 = pointAt(x+dx, y+dy).pos;
    float center_mag_squared = pos_center.x * pos_center.x + pos_center.y*pos_center.y;
    float pos_2_mag_squared = pos_2.x * pos_2.x + pos_2.y*pos_2.y;
    return (pos_2 - pos_center) * (center_mag_squared + pos_2_mag_squared);
}

void main()
{
    int width = 1024;
    int height = 1024;
    int n = int(gl_GlobalInvocationID);
    int x = n % width;
    int y = n / width;

    float ACC_STRENGTH = 10;
    float GRAV_STRENGTH = 0.051;

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

    float SPREAD_STRENGTH = 11.0;
    vec2 neighbor_pull = (spread_received(x, y, 0, 1) + spread_received(x, y, 0, -1) + spread_received(x, y, 1, 0) + spread_received(x, y, -1, 0)) / 4.0;

    vec2 out_pos = in_point.pos + TIMESTEP * (in_point.vel + neighbor_pull*SPREAD_STRENGTH + ACC_STRENGTH*TIMESTEP * acc / 2.0);
    vec2 out_vel = in_point.vel + ACC_STRENGTH * TIMESTEP * (acc + neighbor_pull*0 ) / 2.0;
    //out_vel += neighbor_pos_diff * 1;

    //vec2 out_pos = in_point.pos + neighbor_pos_diff * TIMESTEP * 100;
    //vec2 out_vel = in_point.vel + neighbor_pos_diff * 1.0;
    out_vel *= 0.9999;

    float mag_to_give = received(x,y, 0, 1)
        + received(x,y, 0, -1)
        + received(x,y, 1, 0)
        + received(x,y, -1, 0);

    vec2 neighbor_grav_avg = gravAt(x-1, y)
        + gravAt(x+1, y)
        + gravAt(x, y-1)
        + gravAt(x, y+1);
    neighbor_avg /= 4.0;

    //float pos_mag = sqrt(dot(out_pos, out_pos));
    if(mag_to_give > 0.000000){
        //out_pos *= 1.1;
    }

    //out_pos += mag_to_give * GRAV_STRENGTH;

    //vec2 out_pos = in_point.pos;

    FieldPoint out_point;
    out_point.pos.xy = out_pos;

    Grav.grav[n] += (neighbor_grav_avg - gravAt(x,y)) * TIMESTEP/100;
    Grav.grav[n] += gradAt(x,y) * TIMESTEP/100;
    Grav.grav[n] *= 0.999;
    out_point.vel.xy = out_vel;
    //In.fieldPoints[n].pos = vec2(1.0,1.0);
    //Out.fieldPoints[n].pos = vec2(0.4,0.6);
    Out.fieldPoints[n] = out_point;
}