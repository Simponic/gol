#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

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
  int iterations, log_each_step;
  if (argc == 7) {
    filename = argv[2];
    game.width = atoi(argv[3]);
    game.height = atoi(argv[4]);
    iterations = atoi(argv[5]);
    log_each_step = atoi(argv[6]);
  } else {
    printf("Usage: ./gol simulate <filename | random> <width> <height> <iterations> <log-each-step?1:0>\n");
    filename = "output/out.bin";
    game.height = 10;
    game.width = 10;
    iterations = 5;
    log_each_step = 0;
  }

  clock_t global_start = clock();

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
  clock_t start, end;

  for (int i = 0; i <= iterations; i++) {
    if (i > 0) {
      // Iteration 0 is just the input board
      start = clock();
      next(&game);
      end = clock();
      time_computing_life += ((double) (end - start)) / CLOCKS_PER_SEC;
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
  double total_clock_time = ((double) (clock() - global_start)) / CLOCKS_PER_SEC;
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
