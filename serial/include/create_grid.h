#include <stdio.h>
#include <stdlib.h>
#include "file.h"
#include "game.h"

#ifndef CREATE_GRID_H
#define CREATE_GRID_H

void print_grid(struct GAME* game);
void create_grid(int argc, char** argv);

#endif // CREATE_GRID_H
