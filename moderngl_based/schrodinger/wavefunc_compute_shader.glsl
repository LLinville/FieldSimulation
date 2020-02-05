

layout(local_size_x=GROUP_SIZE) in;

int width = 1024;
int height = 1024;

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
layout(std430, binding=3) buffer pot_in
{
    vec2 pot[];
} Pot;
layout(std430, binding=4) buffer k1_in
{
    vec2 k1[];
} K1;
layout(std430, binding=5) buffer k2_in
{
    vec2 k2[];
} K2;
layout(std430, binding=6) buffer k3_in
{
    vec2 k3[];
} K3;
layout(std430, binding=7) buffer k4_in
{
    vec2 k4[];
} K4;
layout(std430, binding=8) buffer rk_step_in
{
    int rk_step;
} RK_Step;
//
//layout(std430, binding=4) buffer grad_in
//{
//    vec2 grad[];
//} Grad;
//
//layout(std430, binding=5) buffer grav_in
//{
//    vec2 grav[];
//} Grav;

FieldPoint pointAt(int x, int y) {
    if (x <= 0 || x >= 1024 || y <= 0 || y >= 1024) {
        FieldPoint zero;
        zero.pos = vec2(0.0,0.0);
        zero.vel = zero.pos;
        return zero;
    }
    return In.fieldPoints[(y) * width + x];
}

vec2 potAt(int x, int y) {
    if (x <= 0 || x >= 1024 || y <= 0 || y >= 1024) {
        return vec2(0.0, 0.0);
    }
    return Pot.pot[(y) * width + x];
}
vec2 k1At(int x, int y) {
    if (x <= 0 || x >= 1024 || y <= 0 || y >= 1024) {
        return vec2(0.0,0.0);
    }
    return K1.k1[(y) * width + x];
}

vec2 k2At(int x, int y) {
    if (x <= 0 || x >= 1024 || y <= 0 || y >= 1024) {
        return vec2(0.0,0.0);
    }
    return K2.k2[(y) * width + x];
}

vec2 k3At(int x, int y) {
    if (x <= 0 || x >= 1024 || y <= 0 || y >= 1024) {
        return vec2(0.0,0.0);
    }
    return K3.k3[(y) * width + x];
}

vec2 k4At(int x, int y) {
    if (x <= 0 || x >= 1024 || y <= 0 || y >= 1024) {
        return vec2(0.0,0.0);
    }
    return K4.k4[(y) * width + x];
}
//
//vec2 gradAt(int x, int y) {
//    int width = 1024;
//    int height = 1024;
//    return Grad.grad[(y) * width + x];
//}
//
//vec2 gravAt(int x, int y) {
//    int width = 1024;
//    int height = 1024;
//    return Grav.grav[(y) * width + x];
//}

//float received(int x, int y, int dx, int dy) {
//    vec2 avg_grad = (gradAt(x,y) + gradAt(x+dx, y+dy)) / 2;
//    vec2 pos = pointAt(x,y).pos;
//
//    return (sqrt(pos.x*pos.x+pos.y*pos.y))*(avg_grad.x * -1 * dx + avg_grad.y * -1 * dy);
//}
//
//vec2 spread_received(int x, int y, int dx, int dy) {
//    vec2 pos_center = pointAt(x,y).pos;
//    vec2 pos_2 = pointAt(x+dx, y+dy).pos;
//    float center_mag_squared = pos_center.x * pos_center.x + pos_center.y*pos_center.y;
//    float pos_2_mag_squared = pos_2.x * pos_2.x + pos_2.y*pos_2.y;
//    return (pos_2 - pos_center) * (center_mag_squared + pos_2_mag_squared);
//}

void main()
{
    int width = 1024;
    int height = 1024;
    int n = int(gl_GlobalInvocationID);
    int x = n % width;
    int y = n / width;

    FieldPoint in_point = pointAt(x,y);

    float pot = potAt(x,y).x;

//    vec2 neighbor_pos_diff = -4.0 * in_point.pos
//        + pointAt(x-1, y).pos
//        + pointAt(x+1, y).pos
//        + pointAt(x, y-1).pos
//        + pointAt(x, y+1).pos;
//
//    neighbor_pos_diff /= 4.0;
    //neighbor_pos_diff *= 0.0;

    vec2 neighbor_avg = pointAt(x-1, y).pos
        + pointAt(x+1, y).pos
        + pointAt(x, y-1).pos
        + pointAt(x, y+1).pos;
    neighbor_avg /= 4.0;

//    float SPREAD_STRENGTH = 11.0;
//    vec2 neighbor_pull = (spread_received(x, y, 0, 1) + spread_received(x, y, 0, -1) + spread_received(x, y, 1, 0) + spread_received(x, y, -1, 0)) / 4.0;

//    vec2 out_pos = in_point.pos + TIMESTEP * (in_point.vel + neighbor_pull*SPREAD_STRENGTH + POT_STRENGTH*TIMESTEP * pot / 2.0);
//    vec2 out_vel = in_point.vel + POT_STRENGTH * TIMESTEP * (pot + neighbor_pull*0 ) / 2.0;
    //out_vel += neighbor_pos_diff * 1;

    //vec2 out_pos = in_point.pos + neighbor_pos_diff * TIMESTEP * 100;
    //vec2 out_vel = in_point.vel + neighbor_pos_diff * 1.0;
//    out_vel *= 0.9999;

//    float mag_to_give = received(x,y, 0, 1)
//        + received(x,y, 0, -1)
//        + received(x,y, 1, 0)
//        + received(x,y, -1, 0);
//
//    vec2 neighbor_grav_avg = gravAt(x-1, y)
//        + gravAt(x+1, y)
//        + gravAt(x, y-1)
//        + gravAt(x, y+1);
//    neighbor_avg /= 4.0;

    //float pos_mag = sqrt(dot(out_pos, out_pos));
//    if(mag_to_give > 0.000000){
//        //out_pos *= 1.1;
//    }

    //out_pos += mag_to_give * GRAV_STRENGTH;

    vec2 change = (k1At(x, y) + 2.0 * k2At(x, y) + 2.0 * k3At(x, y) + k4At(x, y)) * TIMESTEP / 6.0;



//    vec2 out_pos = in_point.pos + change;
    float neighbor_fade_strength = 0.0005;
    vec2 out_pos = (1.0 - neighbor_fade_strength) * in_point.pos + neighbor_fade_strength * neighbor_avg + change;
    //out_pos = vec2( max(0.0, out_pos.x), max(0.0, out_pos.y));

    FieldPoint out_point;
    out_point.pos.xy = out_pos ;

//    Grav.grav[n] += (neighbor_grav_avg - gravAt(x,y)) * TIMESTEP/100;
//    Grav.grav[n] += gradAt(x,y) * TIMESTEP/100;
//    Grav.grav[n] *= 0.999;
//    out_point.vel.xy = out_vel;
    //In.fieldPoints[n].pos = vec2(1.0,1.0);
    //Out.fieldPoints[n].pos = vec2(0.4,0.6);
    Out.fieldPoints[n] = out_point;
}