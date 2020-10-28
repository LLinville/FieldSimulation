#define BLOCK_SIZE 256
#define SOFTENING 5e-2f
#define N_CHARGES 2


typedef struct {
    float2 *pos,
    float2 *vel,
    float charges[N_CHARGES];
} Particle;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}


__global__
void applyForce(float2 *pos, float2 *vel, float *charges, float *eneg, float *totalBondOrder, float *maxBondOrder, float dt, int *n_array) {

  int n = (int) n_array;

  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  //printf("float size: %d\n", sizeof(float));
  //printf("tID: %d, bID: %d\n", threadIdx.x, blockIdx.x);
//  printf("i=%d: (%.5f, %.5f)\n", i, pos[i].x, pos[i].y);
  dt = 0.0004f;
  if (true) {
    //printf("i=%d: (%f, %f)\n", i, pos[i].x, pos[i].y);
    //d[i].x = dt; d[i].y = n;
    float Fx = 0.0f; float Fy = 0.0f;
    float charge_loss = 0.f;//1.0f + 1.0f * c[i]; // eneg + charge*hardness
    float newTotalBondOrder = 0.f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      int tile_items = tile == gridDim.x-1 ? n%BLOCK_SIZE : BLOCK_SIZE;
      __shared__ float2 spos[BLOCK_SIZE];
//      __shared__ float sc[BLOCK_SIZE];
      __syncthreads();


//      float2 tpos = pos[tile * BLOCK_SIZE + threadIdx.x];
//      if (threadIdx.x < tile_items) {
        spos[threadIdx.x].x = pos[tile * BLOCK_SIZE + threadIdx.x].x;
        spos[threadIdx.x].y = pos[tile * BLOCK_SIZE + threadIdx.x].y;
//        printf("i %d thread %d Set t %d spos %d to (%.5f, %.5f)\n", i, threadIdx.x, tile, threadIdx.x, pos[tile * BLOCK_SIZE + threadIdx.x].x, pos[tile * BLOCK_SIZE + threadIdx.x].y);
//        spos[threadIdx.x].y = tpos.y;
//      } else {
//        printf("items: %d, threadIdx.x: %d\n", tile_items, threadIdx.x);
//      }
      __syncthreads();
//      printf("spos: [%.5f, %.5f, %.5f, %.5f, %.5f]\n", spos[0].y, spos[1].y, spos[2].y, spos[3].y, spos[4].y);
//      if (i >= n) break;
//      if (pos[tile * BLOCK_SIZE + threadIdx.x].x - spos[threadIdx.x].x > 0.001f) {
//        printf("items: %d, threadIdx.x: %d\n", tile_items, threadIdx.x);
//      }

//      printf("Tile %d items: %d\n", tile, tile_items);
      #pragma unroll
      for (int j = 0; i<n && j < BLOCK_SIZE; j++) {
        int tid = tile * BLOCK_SIZE + j;
        if (i == tid || j >= tile_items) continue;


//        if (pos[tid].y - spos[j].y > 0.001f) {
//            printf("Different. tile: %d, tid: %d, j: %d, i: %d, spos: %.5f, p: %.5f\n", tile, tid, j, i, spos[j].y, pos[tid].y);
////            printf("tile: %d, spos[%d]: %.5f, p: %.5f\n", tile, j, spos[j].y, pos[tid].y);
////            printf("spos: [%.5f, %.5f, %.5f, %.5f, %.5f]]\n", spos[0].y, spos[1].y, spos[2].y, spos[3].y, spos[4].y);
//        } else {
//            printf("Correct output, spos: [%.5f, %.5f, %.5f, %.5f, %.5f]\n", spos[0].y, spos[1].y, spos[2].y, spos[3].y, spos[4].y);
//        }
        float dx = spos[j].x - pos[i].x;
        float dy = spos[j].y - pos[i].y;
        //if (dx > 5 || dx < -5 || dy > 5 || dy < -5) continue;

        float dist2 = dx*dx + dy*dy + SOFTENING;

        if (dist2 > 15) {
            //printf("Ignoring %d,%d\n", i, tid);
            //continue;
        }

        float dist = sqrtf(dist2);
        float dist4 = dist2 * dist2;


//        float offsetDist = dist - 0.2f;
//        offsetDist = offsetDist < 0.f ? 0.f : offsetDist;
//        float od2 = offsetDist * offsetDist;
//        float bondOrder = expf(-1.f * od2 * od2* od2* od2* od2* od2);
//        newTotalBondOrder += bondOrder;
//        printf("Bond order %d,%d: %.5f\n", i, tid, bondOrder);

        //if (distSqr > 4) continue;
        float invDist = 1.f/dist;
        float invDist3 = invDist * invDist * invDist;
        float invDist6 = invDist3 * invDist3;
        float strength = 1.0f;
        float fmag = (invDist6 * invDist6*invDist - invDist6*invDist);
//        fmag += 1.1f * c[tid] * c[i];
//        fmag += 1.1f * (4.f * (totalBondOrder[tid] - maxBondOrder[tid]) / (expf(-4.f * (totalBondOrder[tid] - maxBondOrder[tid])) + 1.f)) / dist2;
//        fmag += 0.1f * expf(8.f * (totalBondOrder[tid] - maxBondOrder[tid] - 0.3))/dist2;
        Fx -= dx * invDist * fmag;
        Fy -= dy * invDist * fmag;
        //Fx += dx * invDist3 * strength; Fy += dy * invDist3 * strength;
        //printf("i,j: %d, %d, Fx,Fy:%.5f,%.5f\n", i, j, dx * fmag * strength, dy * fmag * strength);
        //if (tile == gridDim.x-1) {printf("i,j: %d, %d, Dx,Dy:%.5f,%.5f\n", i, j, dx, dy);}

        //charge_loss -= 1.1f*c[tid] / sqrtf(dist2 + 0.1f);
//        charge_loss += (c[i] + eneg[i]*c[i] - c[tid] - c[tid]*eneg[tid]) / sqrtf(dist2 + 0.1f);
      }
      __syncthreads();
    }

    if (i>=n) return;
    float box_width = 50.f / 2.0f;
    //Fx += pos[i].x < -1.f * box_width ? 1.f : 0.f;
    //Fx += pos[i].x > box_width ? -1.f : 0.f;
    //Fy += pos[i].y < -1.f * box_width ? 1.f : 0.f;
    //Fy += pos[i].y > box_width ? -1.f : 0.f;

    vel[i].x *= pos[i].x < -1.f * box_width ? -1.f : 1.f;
    vel[i].x *= pos[i].x > box_width ? -1.f : 1.f;
    vel[i].y *= pos[i].y < -1.f * box_width ? -1.f : 1.f;
    vel[i].y *= pos[i].y > box_width ? -1.f : 1.f;

    //printf("%f\n",vel[i].x);
    vel[i].x = min(max(vel[i].x, -5.f), 5.f);
    vel[i].y = min(max(vel[i].y, -5.f), 5.f);

//    vel[i].x *= 0.9999f;
//    vel[i].y *= 0.9999f;
//    vel[i].x *= 1.000001;
//    vel[i].y *= 1.000001;
//    vel[i].y -= 0.00001f;
    vel[i].x += dt*Fx; vel[i].y += dt*Fy;
    pos[i].x += vel[i].x*dt; pos[i].y += vel[i].y*dt;
    //d[i].y = 2.0f;

    c[i] -= charge_loss * 0.01f;
    totalBondOrder[i] = newTotalBondOrder;

  } else {
    //printf("i out of range: %d\n", i);
  }
}