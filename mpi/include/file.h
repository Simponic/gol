#include "game.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef FILE_H
#define FILE_H

void read_in(char* filename, struct GAME* game); 
void write_out(char* filename, struct GAME* game); 

#endif //FILE_H
