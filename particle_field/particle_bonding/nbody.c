#define BLOCK_SIZE 256
#define SOFTENING 5e-2f

//typedef struct { float2 *pos, float2 *vel; } Particle;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}


__global__
void applyForce(float2 *p, float2 *v, float *c, float dt, int *n_array) {

    //printf("dt: %.5f\n",dt);
    //printf("n: %d\n",n_array);
  int n = (int) n_array;

  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  //printf("float size: %d\n", sizeof(float));
  //printf("tID: %d, bID: %d\n", threadIdx.x, blockIdx.x);
  //printf("i=%d: (%d, %d)\n", i, p[i].x, p[i].y);
  dt = 0.008f;
  //printf("%.12f\n", dt);
  if (i < n) {
    //printf("i=%d: (%f, %f)\n", i, p[i].x, p[i].y);
    //d[i].x = dt; d[i].y = n;
    float Fx = 0.0f; float Fy = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      __shared__ float2 spos[BLOCK_SIZE];
      int tile_items = tile == gridDim.x-1 ? n%BLOCK_SIZE : BLOCK_SIZE;
      float2 tpos = p[tile * BLOCK_SIZE + threadIdx.x];

      spos[threadIdx.x] = tpos;
      __syncthreads();

      //printf("Tile %d items: %d\n", tile, tile_items);
      #pragma unroll
      for (int j = 0; j < BLOCK_SIZE; j++) {
        if (i == j || j >= tile_items) continue;



        float dx = p[tile * BLOCK_SIZE + j].x - p[i].x;
        float dy = p[tile * BLOCK_SIZE + j].y - p[i].y;
        //if (dx > 5 || dx < -5 || dy > 5 || dy < -5) continue;

        float dist2 = dx*dx + dy*dy + SOFTENING;
        float dist = sqrtf(dist2);
        float dist4 = dist2 * dist2;


        float offsetDist = dist - 0.5f;
        offsetDist = offsetDist < 0.f ? 0.f : offsetDist;
        float bondOrder = expf(-1.f * offsetDist * offsetDist);
        printf("Bond order %d,%d: %.5f\n", i, j, bondOrder);

        //if (distSqr > 4) continue;
        float invDist = 1.f/dist;
        float invDist3 = invDist * invDist * invDist;
        float invDist6 = invDist3 * invDist3;
        float strength = 1.0f;
        float fmag = (invDist6 * invDist6*invDist - invDist6*invDist);
        Fx -= dx * invDist * fmag;
        Fy -= dy * invDist * fmag;
        //Fx += dx * invDist3 * strength; Fy += dy * invDist3 * strength;
        //printf("i,j: %d, %d, Fx,Fy:%.5f,%.5f\n", i, j, dx * fmag * strength, dy * fmag * strength);
        //if (tile == gridDim.x-1) {printf("i,j: %d, %d, Dx,Dy:%.5f,%.5f\n", i, j, dx, dy);}
      }
      __syncthreads();
    }

    float box_width = 100.f / 2.0f;
    //Fx += p[i].x < -1.f * box_width ? 1.f : 0.f;
    //Fx += p[i].x > box_width ? -1.f : 0.f;
    //Fy += p[i].y < -1.f * box_width ? 1.f : 0.f;
    //Fy += p[i].y > box_width ? -1.f : 0.f;

    v[i].x *= p[i].x < -1.f * box_width ? -1.f : 1.f;
    v[i].x *= p[i].x > box_width ? -1.f : 1.f;
    v[i].y *= p[i].y < -1.f * box_width ? -1.f : 1.f;
    v[i].y *= p[i].y > box_width ? -1.f : 1.f;

    //printf("%f\n",v[i].x);
    v[i].x = min(max(v[i].x, -1.f), 1.f);
    v[i].y = min(max(v[i].y, -1.f), 1.f);

//    v[i].x *= 0.9999f;
//    v[i].y *= 0.9999f;
    v[i].x *= 1.000001;
    v[i].y *= 1.000001;
    v[i].y -= 0.000001f;
    v[i].x += dt*Fx; v[i].y += dt*Fy;
    p[i].x += v[i].x*dt; p[i].y += v[i].y*dt;
    //d[i].y = 2.0f;

  } else {
    //printf("i out of range: %d\n", i);
  }
}