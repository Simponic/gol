#include "create_grid.h"

// Print entirety of a grid to verify input 
void print_grid(struct GAME* game) {
  printf("\n===GRID===\n");
  for (int y = 0; y < game->height; y++) {
    for (int x = 0; x < game->width; x++) {
      printf("%i ", game->grid[y][x]);
    }
    printf("\n");
  }
}

// Go through user input
void create_grid(int argc, char** argv) {
  char* filename;
  struct GAME game;
  game.padding = 0;
  if (argc == 5) {
    game.width = atoi(argv[2]);
    game.height = atoi(argv[3]);
    filename = argv[4];
  } else {
    printf("Usage: ./gol create-grid <width> <height> <filename>\n");
    exit(1);
  }

  unsigned char** grid = malloc(sizeof(unsigned char*) * game.height);
  for (int y = 0; y < game.height; y++) {
    grid[y] = malloc(sizeof(unsigned char*) * game.width);
    printf("Row %i: ", y);
    for (int x = 0; x < game.width; x++) {
      char temp;
      scanf("%i%c", (unsigned int*)&grid[y][x],&temp);
    }
  }
  game.grid = grid;
  write_out(filename, &game);
  print_grid(&game);
}
