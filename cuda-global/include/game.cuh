#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#ifndef GAME_H
#define GAME_H


struct GAME {
  unsigned char* grid;
  int padding;
  int width;
  int height;
};

__device__ int neighbors(struct GAME game, int x, int y);
__global__ void next(struct GAME game, unsigned char* newGrid);
void randomize(struct GAME* game);

#endif // GAME_H
