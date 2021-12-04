#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "file.h"
#include "game.h"
#include "create_grid.h"


/*
  Rules for life:
  Any live cell with fewer than two live neighbors dies (underpopulation).
  Any live cell with two or three live neighbors continues to live.
  Any live cell with more than three live neighbors dies (overpopulation).
  Any dead cell with exactly three live neighbors becomes a live cell (reproduction).
 */
#define PADDING 10
//#define VERBOSE 1
#define SEED 100

void simulate(int argc, char** argv) {
  srand(SEED);
  char* filename;
  struct GAME game;
  game.padding = PADDING;
  int iterations, log_each_step, threads;
  if (argc == 8) {
    filename = argv[2];
    game.width = atoi(argv[3]);
    game.height = atoi(argv[4]);
    iterations = atoi(argv[5]);
    log_each_step = atoi(argv[6]);
    threads = atoi(argv[7]);
  } else {
    printf("Usage: ./gol simulate <filename | random> <width> <height> <iterations> <log-each-step?1:0> <threads>\n");
    filename = "output/out.bin";
    game.height = 10;
    game.width = 10;
    iterations = 5;
    log_each_step = 0;
    threads = 1;
  }

  double global_start = omp_get_wtime();

  // Allocate space for current grid (1 byte per tile)
  game.grid = malloc(sizeof(unsigned char*) * (game.height+(2*game.padding)));
  for (int i = 0; i < game.height+(2*game.padding); i++) {
    game.grid[i] = malloc(sizeof(unsigned char) * (game.width+(2*game.padding)));
    memset(game.grid[i], 0, game.width+(2*game.padding));
  }

  if (strcmp(filename, "random") == 0) {
    randomize(&game);
  } else {
    read_in(filename, &game);
  }  

  char iteration_file[1024];
  double time_computing_life = 0;
  double start, end;

  for (int i = 0; i <= iterations; i++) {
    if (i > 0) {
      // Iteration 0 is just the input board
      start = omp_get_wtime();
      next(&game, threads);
      end = omp_get_wtime();
      time_computing_life += ((double) (end - start));
    }
    if (log_each_step) {
      #if VERBOSE == 1
      printf("\n===Iteration %i===\n", i);
      for (int y = game.padding; y < game.height+game.padding; y++) {
        for (int x = game.padding; x < game.width+game.padding; x++) {
          printf("%s ", game.grid[y][x] ? "X" : " ");
        }
        printf("\n");
      }
      printf("===End iteration %i===\n", i);
      #endif
      sprintf(iteration_file, "output/iteration-%07d.bin", i);
      write_out(iteration_file, &game);
    }
  }
  double total_clock_time = ((double) (omp_get_wtime() - global_start));
  printf("\n===Timing===\nTime computing life: %f\nClock time: %f\n", time_computing_life, total_clock_time);
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
