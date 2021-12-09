#include "game.h"

int neighbors(struct GAME* game, int x, int y, unsigned char* halo_above, unsigned char* halo_below) {
  int n = 0;

  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (!(dx == 0 && dy == 0) && (x+dx) > 0 && (x+dx) < game->width+(game->padding*2) && (y+dy) < game->height) {
        if (y+dy == -1 && halo_above != NULL) {
          if (halo_above[x+dx]) {
            n++;
          }
        } else if (y+dy == game->height && halo_below != NULL) {
          if (halo_below[x+dx]) {
            n++;
          }
        } else if (game->grid[(y+dy) * (game->width+game->padding*2) + (x+dx)]) {
          n++;
        }
      }
    }
  }
  return n;
}

void next(struct GAME* game, unsigned char* halo_above, unsigned char* halo_below) {
  unsigned char* newGrid = malloc(sizeof(unsigned char) * game->height * (game->width + 2*game->padding));
  for (int y = 0; y < game->height; y++) {
    for (int x = 0; x < game->width + 2*game->padding; x++) {
      int my_neighbors = neighbors(game, x, y, halo_above, halo_below);
      int my_coord = y * (game->width+game->padding*2) + x;
      newGrid[my_coord] = 0; // It's possible that there are artifacts from the last iteration
      if (game->grid[my_coord]) {
        if (my_neighbors < 2 || my_neighbors > 3) {
          newGrid[my_coord] = 0;
        } else  {
          newGrid[my_coord] = 1;
        }
      } else {
        if (my_neighbors == 3) {
          newGrid[my_coord] = 1;
        }
      }
    }
  }
  free(game->grid);
  game->grid = newGrid;
}

void randomize(struct GAME* game) {
  for (int y = game->padding; y < game->height+game->padding; y++) {
    for (int x = game->padding; x < game->width+game->padding; x++) {
      game->grid[y*(game->width+game->padding*2) + x] = (unsigned char) rand() & 1;
    }
  }
}
