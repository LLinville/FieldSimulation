

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

float mag(vec2 vin){
    return sqrt(dot(vin,vin));
}

void main()
{
    int width = 1024;
    int height = 1024;
    int n = int(gl_GlobalInvocationID);
    int x = n % width;
    int y = n / width;

    vec2 grad = vec2(
        (mag(pointAt(x+1, y).pos) - mag(pointAt(x-1, y).pos)) / 2,
        (mag(pointAt(x, y+1).pos) - mag(pointAt(x, y-1).pos)) / 2
    );
    Grad.grad[n] = grad;
}
