
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
layout(std430, binding=3) buffer pot_in
{
    vec2 pot[];
} Pot;

FieldPoint pointAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return In.fieldPoints[(y) * width + x];
}

vec2 potAt(int x, int y) {
    int width = 1024;
    int height = 1024;
    return Pot.pot[(y) * width + x];
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


    float dx = (width/2) - x;
    float dy = (width/2) - y;
    Pot.pot[n] = vec2(1.0 * (dx*dx + dy*dy) / (width * width), 0.0) * 5;
    //Pot.pot[n] = 0;
}
