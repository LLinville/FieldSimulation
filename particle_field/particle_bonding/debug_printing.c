
/*
//typedef struct {
//    float2 *pos,
//    float2 *vel,
//    float charges[N_CHARGES];
//} Particle;
    //printf("float size: %d\n", sizeof(float));
    //printf("tID: %d, bID: %d\n", threadIdx.x, blockIdx.x);
    //  printf("i=%d: (%.5f, %.5f)\n", i, pos[i].x, pos[i].y);
    //printf("i=%d: (%f, %f)\n", i, pos[i].x, pos[i].y);
    //printf("Accessing float indexed %d\n", i * N_CHARGES + chargeIndex);
    //              printf("address %d\n", blockIdx.x * N_CHARGES + chargeIndex);
    //              if (charges[i * N_CHARGES + chargeIndex]*charges[i * N_CHARGES + chargeIndex] > 3.f || scharges[blockIdx.x * N_CHARGES + chargeIndex]*scharges[blockIdx.x * N_CHARGES + chargeIndex] > 3.f) {
    //                 printf("c1, c2: %f, %f\n",
    //                    charges[i * N_CHARGES + chargeIndex],
    //                    scharges[threadIdx.x * N_CHARGES + chargeIndex]);
    //
    //              }

        //      printf("spos: [%.5f, %.5f, %.5f, %.5f, %.5f]\n", spos[0].y, spos[1].y, spos[2].y, spos[3].y, spos[4].y);
    //      if (i >= n) break;
    //      if (pos[tile * BLOCK_SIZE + threadIdx.x].x - spos[threadIdx.x].x > 0.001f) {
    //        printf("items: %d, threadIdx.x: %d\n", tile_items, threadIdx.x);
    //      }

    //      printf("Tile %d items: %d\n", tile, tile_items);

//        if (threadIdx.x < tile_items) {
//            spos[threadIdx.x] = pos[tile * BLOCK_SIZE + threadIdx.x];
//            #pragma unroll
//            for (int chargeIndex = 0; chargeIndex < N_CHARGES; chargeIndex++) {
    //            printf("Accessing float indexed %d\n", i * N_CHARGES + chargeIndex);
    //              printf("address %d\n", blockIdx.x * N_CHARGES + chargeIndex);
    //              if (charges[i * N_CHARGES + chargeIndex]*charges[i * N_CHARGES + chargeIndex] > 3.f || scharges[blockIdx.x * N_CHARGES + chargeIndex]*scharges[blockIdx.x * N_CHARGES + chargeIndex] > 3.f) {
    //                 printf("c1, c2: %f, %f\n",
    //                    charges[i * N_CHARGES + chargeIndex],
    //                    scharges[threadIdx.x * N_CHARGES + chargeIndex]);
    //
    //              }
//                scharges[threadIdx.x * N_CHARGES + chargeIndex] = charges[(tile * BLOCK_SIZE + threadIdx.x) * N_CHARGES + chargeIndex];
//            }
//        }
    //        spos[threadIdx.x].y = pos[tile * BLOCK_SIZE + threadIdx.x].y;
    //        printf("i %d thread %d Set t %d spos %d to (%.5f, %.5f)\n", i, threadIdx.x, tile, threadIdx.x, pos[tile * BLOCK_SIZE + threadIdx.x].x, pos[tile * BLOCK_SIZE + threadIdx.x].y);
    //        spos[threadIdx.x].y = tpos.y;
    //      } else {
    //        printf("items: %d, threadIdx.x: %d\n", tile_items, threadIdx.x);
    //      }
    */


    //        if (pos[tid].y - spos[j].y > 0.001f) {
    //            printf("Different. tile: %d, tid: %d, j: %d, i: %d, spos: %.5f, p: %.5f\n", tile, tid, j, i, spos[j].y, pos[tid].y);
    ////            printf("tile: %d, spos[%d]: %.5f, p: %.5f\n", tile, j, spos[j].y, pos[tid].y);
    ////            printf("spos: [%.5f, %.5f, %.5f, %.5f, %.5f]]\n", spos[0].y, spos[1].y, spos[2].y, spos[3].y, spos[4].y);
    //        } else {
    //            printf("Correct output, spos: [%.5f, %.5f, %.5f, %.5f, %.5f]\n", spos[0].y, spos[1].y, spos[2].y, spos[3].y, spos[4].y);
    //        }



    //            if (chargePullMag * chargePullMag > 1) {
//                printf("i,j: %d, %d | c1, c2, cAtt, dist2: %f, %f, %f, %f\n", i, j,
//                        charges[i * N_CHARGES + chargeIndex],
//                        scharges[j * N_CHARGES + chargeIndex],
//                        chargeAttractions[i * N_CHARGES + chargeIndex],
//                        dist2);
//            }

//printf("i,j: %d, %d, Fx,Fy:%.5f,%.5f\n", i, j, dx * fmag * strength, dy * fmag * strength);
        //if (tile == gridDim.x-1) {printf("i,j: %d, %d, Dx,Dy:%.5f,%.5f\n", i, j, dx, dy);}





























