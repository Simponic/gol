#include "game.cuh"

// Count the number of life neighbors a cell has
__device__ int neighbors(struct GAME game, int x, int y) {
  int n = 0;

  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (!(dx == 0 && dy == 0) && (x+dx) > 0 && (y+dy) > 0 && (x+dx) < game.width+(game.padding*2) && (y+dy) < game.height+(game.padding*2)) {
        if (game.grid[(y+dy) * (game.width+game.padding*2) + (x+dx)]) {
          n++;
        }
      }
    }
  }
  return n;
}

// Compute the next iteration of a board
// We have to give it the newGrid as a parameter otherwise 
// each block will be computing its own version of the next grid
__global__ void next(struct GAME game, unsigned char* newGrid) {
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idy < game.height+game.padding*2 && idx < game.width+game.padding*2) {
    int my_neighbors = neighbors(game, idx, idy);
    int my_coord = idy * (game.width+game.padding*2) + idx;
    newGrid[my_coord] = 0; // It's possible that there are artifacts from the last iteration
    if (game.grid[my_coord]) {
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

// Randomly assign life value to each cell
void randomize(struct GAME* game) {
  for (int y = game->padding; y < game->height+game->padding; y++) {
    for (int x = game->padding; x < game->width+game->padding; x++) {
      game->grid[y*(game->width+game->padding*2) + x] = (unsigned char) rand() & 1;
    }
  }
}
