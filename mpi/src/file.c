#include "file.h"

void read_in(char* filename, struct GAME* game) {
  FILE* file = fopen(filename, "rb");
  for (int i = game->padding; i < game->height+game->padding; i++) {
    fread(&game->grid[i*(game->width + 2*game->padding) + game->padding], sizeof(unsigned char), game->width, file);
  }
  fclose(file);
}

void write_out(char* filename, struct GAME* game) {
  FILE* file = fopen(filename, "w+");
  for (int i = game->padding; i < game->height+game->padding; i++) {
    fwrite(&game->grid[i*(game->width + 2*game->padding) + game->padding], sizeof(unsigned char), game->width, file);
  }
  fclose(file);
}
