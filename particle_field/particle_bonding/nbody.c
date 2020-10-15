#define BLOCK_SIZE 256
#define SOFTENING 5e-2f

//typedef struct { float2 *pos, float2 *vel; } Particle;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}



__global__
void applyForce(float2 *p, float2 *v, float dt, int *n_array) {

    //printf("dt: %.5f\n",dt);
    //printf("n: %d\n",n_array);
  int n = (int) n_array;

  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  //printf("float size: %d\n", sizeof(float));
  //printf("tID: %d, bID: %d\n", threadIdx.x, blockIdx.x);
  //printf("i=%d: (%d, %d)\n", i, p[i].x, p[i].y);
  dt = 0.01f;
  //printf("%.12f\n", dt);
  if (i < n) {
    //printf("i=%d: (%f, %f)\n", i, p[i].x, p[i].y);
    //d[i].x = dt; d[i].y = n;
    float Fx = 0.0f; float Fy = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      __shared__ float2 spos[BLOCK_SIZE];
      //int tid=tile * blockDim.x + threadIdx.x;
      float2 tpos = p[tile * BLOCK_SIZE + threadIdx.x];

      spos[threadIdx.x] = tpos;//make_float2(tpos.x, tpos.y);
      __syncthreads();

      int tile_items = tile == gridDim.x-1 ? n%BLOCK_SIZE : BLOCK_SIZE;
      //printf("Tile %d items: %d\n", tile, tile_items);
      #pragma unroll
      for (int j = 0; j < tile_items; j++) {

        if (i == j) continue;
//        if (tile == gridDim.x-1 && j>=n%BLOCK_SIZE) {
//        //printf("Breaking on tid=%d\n",tid);
//        //printf("tID: %d, j: %d\n", tile, j);
//        continue;
//        } else {
//            //printf("calculating tid=%d\n",tid);
//        }


        float dx = spos[j].x - p[i].x;
        float dy = spos[j].y - p[i].y;
        float distSqr = dx*dx + dy*dy + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float invDist6 = invDist3 * invDist3;
        float strength = 1.0f;
        float fmag = (invDist6 * invDist6*invDist - invDist6*invDist);
        Fx -= dx * invDist * fmag;
        Fy -= dy * invDist * fmag;
        //Fx += dx * invDist3 * strength; Fy += dy * invDist3 * strength;
        //printf("i,j: %d, %d, Fx,Fy:%.5f,%.5f\n", i, tid, dx * fmag * strength, dy * fmag * strength);
        //printf("i,j: %d, %d, Dx,Dy:%.5f,%.5f\n", i, j, dx, dy);
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
    //v[i].x = min(max(v[i].x, -1.f), 1.f);
    //v[i].y = min(max(v[i].y, -1.f), 1.f);

    v[i].x *= 0.999999f;
    v[i].y *= 0.999999f;
    v[i].x += dt*Fx; v[i].y += dt*Fy;
    p[i].x += v[i].x*dt; p[i].y += v[i].y*dt;
    //d[i].y = 2.0f;

  } else {
    //printf("i out of range: %d\n", i);
  }
}