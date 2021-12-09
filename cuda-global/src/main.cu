#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <cstring>

#include "file.cuh"
#include "game.cuh"
#include "create_grid.cuh"


/*
  Rules for life:
  Any live cell with fewer than two live neighbors dies (underpopulation).
  Any live cell with two or three live neighbors continues to live.
  Any live cell with more than three live neighbors dies (overpopulation).
  Any dead cell with exactly three live neighbors becomes a live cell (reproduction).
 */
#define BLOCK 32
#define PADDING 10
//#define VERBOSE 1
#define SEED 100

// gpuErrchk source: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

// Do the simulation
void simulate(int argc, char** argv) {
  srand(SEED);
  cudaEvent_t global_start, global_end;
  cudaEventCreate(&global_start);
  cudaEventCreate(&global_end);
  cudaEventRecord(global_start);
  char* filename;
  struct GAME game;
  game.padding = PADDING;
  int iterations, log_each_step;
  if (argc == 7) {
    // Parse the arguments
    filename = argv[2];
    game.width = atoi(argv[3]);
    game.height = atoi(argv[4]);
    iterations = atoi(argv[5]);
    log_each_step = atoi(argv[6]);
  } else {
    printf("Usage: ./gol simulate <filename | random> <width> <height> <iterations> <log-each-step?1:0>\n");
    filename = "random";
    game.height = 10;
    game.width = 10;
    iterations = 5;
    log_each_step = 0;
  }

  // Allocate space for current grid (1 byte per tile)
  int size = (game.height+(2*game.padding)) * (game.width+(2*game.padding)) * sizeof(unsigned char);
  game.grid = (unsigned char*)malloc(size);
  memset(game.grid, 0, size);

  // Choose where to read initial position
  if (strcmp(filename, "random") == 0) {
    randomize(&game);
  } else {
    read_in(filename, &game);
  }  

  char iteration_file[1024];

  // Allocate device memory
  unsigned char* grid_d;
  unsigned char* newGrid;
  gpuErrchk(cudaMalloc(&grid_d, size));
  gpuErrchk(cudaMalloc(&newGrid, size));
  gpuErrchk(cudaMemcpy(grid_d, game.grid, size, cudaMemcpyHostToDevice)); // Copy the initial grid to the device
  free(game.grid);
  game.grid = grid_d; // Use the device copy

  // The grid that we will copy results 
  unsigned char* grid_h = (unsigned char*)malloc(size);
  unsigned char* temp;

  // Calculate grid width for kernel
  int grid_width = (int)ceil((game.width+(2*game.padding))/(float)BLOCK);
  int grid_height = (int)ceil((game.height+(2*game.padding))/(float)BLOCK);
  dim3 dim_grid(grid_width, grid_height, 1);
  dim3 dim_block(BLOCK, BLOCK, 1);

  // Timing
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  double time_computing_life = 0;
  float local_time = 0;

  for (int i = 0; i <= iterations; i++) {
    // Iteration 0 will just be the initial grid
    if (i > 0) {
      cudaEventRecord(start);
      // Compute the next grid
      next<<<dim_grid, dim_block>>>(game, newGrid);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&local_time, start, end);
      time_computing_life += local_time/1000;

      // Swap game.grid and newGrid
      temp = game.grid;
      game.grid = newGrid;
      newGrid = temp;
    }
    if (log_each_step) {
      // If we are logging each step, perform IO operations
      gpuErrchk(cudaMemcpy(grid_h, game.grid, size, cudaMemcpyDeviceToHost));
      #ifdef VERBOSE
        // Print the board without the padding elements
        printf("\n===Iteration %i===\n", i);
        for (int y = game.padding; y < game.height+game.padding; y++) {
          for (int x = game.padding; x < game.width+game.padding; x++) {
            printf("%s ", grid_h[y*(game.width+2*game.padding) + x] ? "X" : " ");
          }
          printf("\n");
        }
        printf("===End iteration %i===\n", i);
      #endif
      // Save to a file
      sprintf(iteration_file, "output/iteration-%07d.bin", i);
      temp = game.grid;
      game.grid = grid_h;
      write_out(iteration_file, &game);
      game.grid = temp;
    }
  }
  cudaEventRecord(global_end);
  cudaEventSynchronize(global_end);
  float global_time;
  cudaEventElapsedTime(&global_time, global_start, global_end);

  printf("\n===Timing===\nTime computing life: %f\nClock time: %f\n", time_computing_life, global_time/(double)1000);
}

int main(int argc, char** argv) {
  if (argc >= 2) {
    if (strcmp(argv[1], "simulate") == 0) {
      simulate(argc, argv);
    } else if (strcmp(argv[1], "create-grid") == 0) {
      create_grid(argc, argv);
    } else {
      printf("Unknown input: %s\n", argv[1]);
      exit(1);
    }
  } else {
    printf("Usage: ./gol <simulate | create-grid>\n");
    exit(1);
  }
  return 0;
}
