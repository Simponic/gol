.DEFAULT_GOAL := all

INCLUDES = -I include/
FLAGS= -std=c99

game.o: include/game.h src/game.c
	mpicc -c src/game.c $(INCLUDES) $(FLAGS) -o build/game.o

file.o: game.o include/file.h src/file.c
	mpicc -c src/file.c $(INCLUDES) $(FLAGS) -o build/file.o

create_grid.o: file.o game.o include/create_grid.h src/create_grid.c
	mpicc -c src/create_grid.c $(INCLUDES) $(FLAGS) -o build/create_grid.o

gol: game.o file.o create_grid.o
	mpicc $(INCLUDES) $(FLAGS) -o gol build/game.o build/file.o build/create_grid.o src/main.c

clean:
	$(RM) build/*.o gol output/*.bin

all: gol
