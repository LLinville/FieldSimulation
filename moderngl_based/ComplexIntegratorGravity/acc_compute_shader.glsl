
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

void main()
{
    int width = 1024;
    int height = 1024;
    int n = int(gl_GlobalInvocationID);
    int x = n % width;
    int y = n / width;

    FieldPoint in_point = pointAt(x,y);

    vec2 neighbor_pos_diff = -4.0 * in_point.pos
        + pointAt(x-1, y).pos
        + pointAt(x+1, y).pos
        + pointAt(x, y-1).pos
        + pointAt(x, y+1).pos;

    Acc.acc[n] = 0*neighbor_pos_diff / 4.0;
}
