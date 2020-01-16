

layout(local_size_x=GROUP_SIZE) in;

uniform int rk_target_buffer;
int width = 1024;
int height = 1024;

struct FieldPoint {
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
} RK_Step_in;

layout(std430, binding=9) buffer rk_step_out
{
    int rk_step;
} RK_Step_out;

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

float mag(vec2 vin){
    return sqrt(dot(vin,vin));
}

void main()
{
    int n = int(gl_GlobalInvocationID);
    int x = n % width;
    int y = n / width;

    vec2 center = pointAt(x, y).pos;
    vec2 left = pointAt(x-1, y).pos;
    vec2 right = pointAt(x+1, y).pos;
    vec2 up = pointAt(x, y-1).pos;
    vec2 down = pointAt(x, y+1).pos;

    int rk_target_buffer = RK_Step_in.rk_step;

    if (rk_target_buffer == 1) {
        vec2 hamiltonian = (right + left + up + down - 4.0*center) / 4.0 - potAt(x,y).x * center;
        K1.k1[n] = vec2( -1.0 * hamiltonian.y, hamiltonian.x);
        //RK_Step.rk_step = 2;
    }
    else if (rk_target_buffer == 2) {
        center += k1At(x, y) * TIMESTEP / 2;
        vec2 hamiltonian = (right + left + up + down - 4.0*center) / 4.0 - potAt(x,y).x * center;
        K2.k2[n] = vec2( -1.0 * hamiltonian.y, hamiltonian.x);
        //RK_Step.rk_step = 3;
    }
    else if (rk_target_buffer == 3) {
        center += k2At(x, y) * TIMESTEP / 2;
        vec2 hamiltonian = (right + left + up + down - 4.0*center) / 4.0 - potAt(x,y).x * center;
        K3.k3[n] = vec2( -1.0 * hamiltonian.y, hamiltonian.x);
        //RK_Step.rk_step = 4;
    }
    else if (rk_target_buffer == 4) {
        center += k3At(x, y) * TIMESTEP;
        vec2 hamiltonian = (right + left + up + down - 4.0*center) / 4.0 - potAt(x,y).x * center;
        K4.k4[n] = vec2( -1.0 * hamiltonian.y, hamiltonian.x);
        //RK_Step.rk_step = 1;
    }

    if (n == 0) {
        // Chosen to update RK step
        if (RK_Step_in.rk_step == 4){
            RK_Step_out.rk_step = 1;
        } else {
            RK_Step_out.rk_step = RK_Step_in.rk_step + 1;
        }
    }

//    vec2 grad = vec2(
//        (mag(pointAt(x+1, y).pos) - mag(pointAt(x-1, y).pos)) / 2,
//        (mag(pointAt(x, y+1).pos) - mag(pointAt(x, y-1).pos)) / 2
//    );
//    Grad.grad[n] = grad;
}
