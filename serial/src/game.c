#include "game.h"

int neighbors(struct GAME* game, int x, int y) {
  int n = 0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (!(dx == 0 && dy == 0) && (x+dx) > 0 && (y+dy) > 0 && (x+dx) < game->width+(game->padding*2) && (y+dy) < game->height+(game->padding*2)) {
        if (game->grid[y+dy][x+dx]) {
          n++;
        }
      }
    }
  }

  return n;
}

void next(struct GAME* game) {
  unsigned char** newGrid = malloc(sizeof(unsigned char*) * (game->height+(game->padding*2)));
  int size = sizeof(unsigned char) * (game->width+(game->padding*2));
  for (int y = 0; y < game->height+(game->padding*2); y++) {
    newGrid[y] = malloc(size);
    memset(newGrid[y], 0, size);
  }

  for (int y = 0; y < game->height+(game->padding*2); y++) {
    for (int x = 0; x < game->width+(game->padding*2); x++) {
      int my_neighbors = neighbors(game, x, y);
      if (game->grid[y][x]) {
        if (my_neighbors < 2 || my_neighbors > 3) {
          newGrid[y][x] = 0;
        } else  {
          newGrid[y][x] = 1;
        }
      } else {
        if (my_neighbors == 3) {
          newGrid[y][x] = 1;
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
      game->grid[y][x] = rand() & 1;
    }
  }
}
