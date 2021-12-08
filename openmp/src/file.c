#include "file.h"

// Read a grid from a binary file into the space without padding
void read_in(char* filename, struct GAME* game) {
  FILE* file = fopen(filename, "rb");
  for (int i = game->padding; i < game->height+game->padding; i++) {
    fread(game->grid[i] + game->padding, sizeof(unsigned char), game->width, file);
  }
  fclose(file);
}

// Write a grid to a binary file into the space without padding
void write_out(char* filename, struct GAME* game) {
  FILE* file = fopen(filename, "w+");
  for (int i = game->padding; i < game->height+game->padding; i++) {
    fwrite(game->grid[i] + game->padding, sizeof(unsigned char), game->width, file);
  }
  fclose(file);
}
