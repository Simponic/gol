#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#ifndef GAME_H
#define GAME_H


struct GAME {
  unsigned char** grid;
  int padding;
  int width;
  int height;
};

int neighbors(struct GAME* game, int x, int y);
void next(struct GAME* game, int num_threads);
void update(struct GAME* game, int x1, int x2, int y1, int y2);
void randomize(struct GAME* game);

#endif // GAME_H
