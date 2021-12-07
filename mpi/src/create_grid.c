#include "create_grid.h"

void print_grid(struct GAME* game) {
  printf("\n===GRID===\n");
  for (int y = 0; y < game->height; y++) {
    for (int x = 0; x < game->width; x++) {
      printf("%i ", game->grid[y*(game->width+game->padding*2) + x]);
    }
    printf("\n");
  }
}

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

  int size = (game.width+game.padding*2) * (game.height+game.padding*2);
  unsigned char* grid = (unsigned char*)malloc(sizeof(unsigned char) * size);
  for (int y = 0; y < game.height; y++) {
    printf("Row %i: ", y);
    for (int x = 0; x < game.width; x++) {
      char temp;
      scanf("%i%c", (unsigned int*)&game.grid[y*(game.width+game.padding*2) + x],&temp);
    }
  }
  game.grid = grid;
  write_out(filename, &game);
  print_grid(&game);
}
