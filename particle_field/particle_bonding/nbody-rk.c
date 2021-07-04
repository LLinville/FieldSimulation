#define BLOCK_SIZE 256
#define SOFTENING 5e-2f
#define N_CHARGES 3


void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__device__
float reflection_force_down(float uPos, float uBound) {
    float depth = uBound - uPos;
    depth = depth > 0.f ? 0.f : depth;
    return -1.f * depth * depth;
}

__device__
float reflection_force_up(float uPos, float uBound) {
    float depth = uPos - uBound;
    depth = depth > 0.f ? 0.f : depth;
    return 1.f * depth * depth;
}










__global__
void applyForce(float2 *pos, float2 *vel, float *charges, float *chargeAttractions, float *eneg, float *totalBondOrder, float *maxBondOrder, float dt, int *n_array) {

    int n = (int) n_array;

    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;


    dt = 0.0234f;
    float Fx = 0.0f; float Fy = 0.0f;
    float charge_loss = 0.f;//1.0f + 1.0f * c[i]; // eneg + charge*hardness
    float newTotalBondOrder = 0.f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      int tile_items = tile == gridDim.x-1 ? n%BLOCK_SIZE : BLOCK_SIZE;
      __shared__ float2 spos[BLOCK_SIZE];
      __shared__ float scharges[BLOCK_SIZE * N_CHARGES];
      __syncthreads();
      if (threadIdx.x < tile_items) {
        spos[threadIdx.x] = pos[tile * BLOCK_SIZE + threadIdx.x];
        #pragma unroll
        for (int chargeIndex = 0; chargeIndex < N_CHARGES; chargeIndex++) {
//            printf("Accessing float indexed %d\n", i * N_CHARGES + chargeIndex);
//              printf("address %d\n", blockIdx.x * N_CHARGES + chargeIndex);
//              if (charges[i * N_CHARGES + chargeIndex]*charges[i * N_CHARGES + chargeIndex] > 3.f || scharges[blockIdx.x * N_CHARGES + chargeIndex]*scharges[blockIdx.x * N_CHARGES + chargeIndex] > 3.f) {
////                 printf("c1, c2: %f, %f\n",
//                    charges[i * N_CHARGES + chargeIndex],
//                    scharges[threadIdx.x * N_CHARGES + chargeIndex]);
//
//              }
            scharges[threadIdx.x * N_CHARGES + chargeIndex] = charges[(tile * BLOCK_SIZE + threadIdx.x) * N_CHARGES + chargeIndex];
        }
      }
      __syncthreads();



//      #pragma unroll
      for (int j = 0; i<n && j < BLOCK_SIZE; j++) {
        int tid = tile * BLOCK_SIZE + j;
        if (i == tid || j >= tile_items) continue;

        float dx = spos[j].x - pos[i].x;
        float dy = spos[j].y - pos[i].y;

        float dist2 = dx*dx + dy*dy + SOFTENING;


//        if (dist2 > 5) {
////            printf("Ignoring %d,%d\n", i, tid);
//            continue;
//        }

        float dist = sqrtf(dist2);
        float dist4 = dist2 * dist2;


//        float offsetDist = dist - 0.2f;
//        offsetDist = offsetDist < 0.f ? 0.f : offsetDist;
//        float od2 = offsetDist * offsetDist;
//        float bondOrder = expf(-1.f * od2 * od2* od2* od2* od2* od2);
//        newTotalBondOrder += bondOrder;
//            printf("Bond order %d,%d: %.5f\n", i, tid, bondOrder);

        if (dist2 > 16) continue;
        float invDist = 1.f/dist;
        float invDist3 = invDist * invDist * invDist;
        float invDist6 = invDist3 * invDist3;
        float lj_strength = 0.9990f;
        float c_strength = 1.f*0.00510f;
        float fmag = lj_strength * (invDist6 * invDist6*invDist - invDist6*invDist);

//        printf("i,j: %d, %d, Fx,Fy:%.5f,%.5f\n", i, j, dx * fmag * lj_strength, dy * fmag * lj_strength);

//        #pragma unroll
//        for (int chargeIndex = 0; chargeIndex < N_CHARGES; chargeIndex++) {
//            float chargePullMag = -1.f * (c_strength *
//                1.f*//charges[i * N_CHARGES + chargeIndex] *
//                scharges[j * N_CHARGES + chargeIndex] *
//                chargeAttractions[i * N_CHARGES + chargeIndex]) /
//                (dist2+0.11);
//
////            float chargePullMag = 0.f;
//            fmag += chargePullMag;
//        }
//            fmag += 1.1f * c[tid] * c[i];
//        fmag += 1.1f * (4.f * (totalBondOrder[tid] - maxBondOrder[tid]) / (expf(-4.f * (totalBondOrder[tid] - maxBondOrder[tid])) + 1.f)) / dist2;
//            fmag += 0.1f * expf(1.f * (totalBondOrder[tid] - maxBondOrder[tid] - 0.3))/dist2;
        Fx -= dx * invDist * fmag;
        Fy -= dy * invDist * fmag;
        //Fx += dx * invDist3 * strength; Fy += dy * invDist3 * strength;


        //charge_loss -= 1.1f*c[tid] / sqrtf(dist2 + 0.1f);
    //        charge_loss += (c[i] + eneg[i]*c[i] - c[tid] - c[tid]*eneg[tid]) / sqrtf(dist2 + 0.1f);
      }
      __syncthreads();
    }

    if (i>=n) return;
    float box_width = 150.f / 2.0f;
    //Fx += pos[i].x < -1.f * box_width ? 1.f : 0.f;
    //Fx += pos[i].x > box_width ? -1.f : 0.f;
//    Fy += pos[i].y < -1.f * box_width ? 1.f : 0.f;
    //Fy += pos[i].y > box_width ? -1.f : 0.f;

//    vel[i].x *= pos[i].x < -1.f * box_width ? -1.f : 1.f;
//    vel[i].x *= pos[i].x > box_width ? -1.f : 1.f;
//    vel[i].y *= pos[i].y < -1.f * box_width ? -1.f : 1.f;
//    vel[i].y *= pos[i].y > box_width ? -1.f : 1.f;
    vel[i].y += 0.01f * reflection_force_up(pos[i].y, -1.f * box_width);
    vel[i].x += 0.01f * reflection_force_up(pos[i].x, -1.f * box_width);
    vel[i].y += 0.01f * reflection_force_down(pos[i].y, 1.f * box_width);
    vel[i].x += 0.01f * reflection_force_down(pos[i].x, 1.f * box_width);


//    pos[i].x = min(max(pos[i].x, -1.f * box_width), box_width);
//    pos[i].y = min(max(pos[i].y, -1.f * box_width), box_width);

//    printf("%f\n",vel[i].x);
    vel[i].x = min(max(vel[i].x, -3.f), 3.f);
    vel[i].y = min(max(vel[i].y, -3.f), 3.f);


    vel[i].x *= 0.9999f;
    vel[i].y *= 0.9999f;
//    vel[i].x *= 1.000001;
//    vel[i].y *= 1.000001;
//    vel[i].y -= 0.000251f;
    vel[i].x += dt*Fx; vel[i].y += dt*Fy;
    pos[i].x += vel[i].x*dt; pos[i].y += vel[i].y*dt;



//    charges[i] -= charge_loss * 0.01f;
//    printf("bond order for %d: %f\n", i, newTotalBondOrder);
    totalBondOrder[i] = newTotalBondOrder;


}