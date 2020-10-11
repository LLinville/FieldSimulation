#define BLOCK_SIZE 256
#define SOFTENING 1e-5f

typedef struct { float2 *pos, float2 *vel; } Particle;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void applyForce(float2 *p, float2 *v, float2 *d, float dt, int n) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  printf("float size: %d\n", sizeof(float));
  //printf("bID: %d, tID: %d\n", blockIdx.x, threadIdx.x);
  //printf("i=%d: (%d, %d)\n", i, p[i].x, p[i].y);
  dt = 0.00001f;
  if (i < n) {
    //printf("i=%d: (%d, %d)\n", i, p[i].x, p[i].y);
    d[i].x = dt; d[i].y = n;
    float Fx = 0.0f; float Fy = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      __shared__ float2 spos[BLOCK_SIZE];
      float2 tpos = p[tile * blockDim.x + threadIdx.x];
      spos[threadIdx.x] = make_float2(tpos.x, tpos.y);
      __syncthreads();

      #pragma unroll
      for (int j = 0; j < BLOCK_SIZE; j++) {
        //printf("i,j: %d, %d\n", i, j);
        float dx = spos[j].x - p[i].x;
        float dy = spos[j].y - p[i].y;
        float distSqr = dx*dx + dy*dy + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float strength = 1.0f;

        Fx += dx * invDist3 * strength; Fy += dy * invDist3 * strength;
      }
      __syncthreads();
    }

    //v[i].x += 1.5f;
    v[i].x += dt*Fx; v[i].y += dt*Fy;
    p[i].x += v[i].x*dt; p[i].y += v[i].y*dt;
    //d[i].y = 2.0f;

  }
}