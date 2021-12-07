#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef GAME_H
#define GAME_H

struct GAME {
  unsigned char* grid;
  int padding;
  int width;
  int height;
};

int neighbors(struct GAME* game, int x, int y, unsigned char* halo_above, unsigned char* halo_below);
void next(struct GAME* game, unsigned char* halo_above, unsigned char* halo_below);
void randomize(struct GAME* game);

#endif // GAME_H
