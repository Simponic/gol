#include "game.h"

// Calculate the number of live neighbors a cell has
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

// Compute the next iteration of a board
void next(struct GAME* game, int threads) {
  unsigned char** newGrid = malloc(sizeof(unsigned char*) * (game->height+(game->padding*2)));
  int y,x,i,size;
  size = sizeof(unsigned char) * (game->width+(game->padding*2));
  for (y = 0; y < game->height+(game->padding*2); y++) {
    newGrid[y] = malloc(size);
    memset(newGrid[y], 0, size);
  }
  int total_width = game->width+(game->padding*2);
  int total_height = game->height+(game->padding*2);

  int per_thread = (total_width * total_height) / threads;

#pragma omp parallel num_threads(threads) shared(per_thread, threads, total_width, total_height, newGrid, game) private(y,x,i)
  {
    // Each thread gets a number of cells to compute
    int me = omp_get_thread_num();
    int thread_start = per_thread * me;
    int thread_end = thread_start + per_thread + (me == threads-1 ? (total_width*total_height) % per_thread : 0);
    for (i = thread_start; i < thread_end; i++) {
      // Iterate through each cell assigned for this thread
      y = i / total_width;
      x = i % total_width;
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

//Rnadomly assign life value to each cell
void randomize(struct GAME* game) {
  for (int y = game->padding; y < game->height+game->padding; y++) {
    for (int x = game->padding; x < game->width+game->padding; x++) {
      game->grid[y][x] = rand() & 1;
    }
  }
}
