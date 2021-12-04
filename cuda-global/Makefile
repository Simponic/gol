.DEFAULT_GOAL := all

INCLUDES = -I include/

game.o: include/game.cuh src/game.cu
	nvcc -c src/game.cu $(INCLUDES) -o build/game.o

file.o: game.o include/file.cuh src/file.cu
	nvcc -c src/file.cu $(INCLUDES) -o build/file.o

create_grid.o: file.o game.o include/create_grid.cuh src/create_grid.cu
	nvcc -c src/create_grid.cu $(INCLUDES) -o build/create_grid.o

gol: game.o file.o create_grid.o
	nvcc $(INCLUDES) -o gol build/game.o build/file.o build/create_grid.o src/main.cu

clean:
	$(RM) build/*.o gol output/*.bin

all: gol
