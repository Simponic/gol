#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

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

// A structure to keep the global arguments because each process
// will use its own GAME structure
struct Args {
  int process_count;
  int iterations;
  int log_each_step;
  int width;
  int height;
  int padding;
  int rows_per_proc;
  int data_per_proc;
};

// Make a datatype out of an Args struct
void broadcast_and_receive_input(MPI_Comm comm, struct Args* args) {
  int blocks[8] = {1,1,1,1,1,1,1,1};
  MPI_Aint displacements[8];
  MPI_Datatype types[8] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
  MPI_Datatype arg_t;

  displacements[0] = offsetof(struct Args, process_count);
  displacements[1] = offsetof(struct Args, iterations);
  displacements[2] = offsetof(struct Args, log_each_step);
  displacements[3] = offsetof(struct Args, width);
  displacements[4] = offsetof(struct Args, height);
  displacements[5] = offsetof(struct Args, padding);
  displacements[6] = offsetof(struct Args, rows_per_proc);
  displacements[7] = offsetof(struct Args, data_per_proc);

  MPI_Type_create_struct(8, blocks, displacements, types, &arg_t);
  MPI_Type_commit(&arg_t);
  MPI_Bcast(args, 1, arg_t, 0, comm);
}

// Scatter the grid among nodes
void scatter_data(MPI_Comm comm, struct Args* args, unsigned char* local_data, int rank, int* data_counts, int* displacements, char* filename) {
  unsigned char* data;

  int grid_size = (args->height + args->padding*2)*(args->width + args->padding*2);
  if (rank == 0) {
    struct GAME game;
    game.width = args->width;
    game.height = args->height;
    game.padding = args->padding;
    int size = sizeof(unsigned char)*grid_size;
    data = malloc(size);
    memset(data, 0, size); 
    game.grid = data;
    // Choose where to read initial position
    if (strcmp(filename, "random") == 0) {
      randomize(&game);
    } else {
      read_in(filename, &game);
    }
  }
  // Do the scatter (some nodes may work on more rows)
  MPI_Scatterv(data, data_counts, displacements, MPI_UNSIGNED_CHAR, local_data, data_counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

  if (rank == 0) {
    free(data);
  }
}


// Do the simulation
void simulate(int argc, char** argv) {
  srand(SEED);
  struct Args args;
  args.padding = PADDING;

  // Initialize MPI stuff
  int rank, process_count;
  MPI_Comm comm;
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &args.process_count);

  char* filename;
  double global_start;
  if (rank == 0) {
    // Parse the arguments
    if (argc == 7) {
      filename = argv[2];
      args.width = atoi(argv[3]);
      args.height = atoi(argv[4]);
      args.iterations = atoi(argv[5]);
      args.log_each_step = atoi(argv[6]);
    } else {
      printf("Usage: ./gol simulate <filename | random> <width> <height> <iterations> <log-each-step?1:0>\n");
      filename = "random";
      args.height = 5;
      args.width = 5;
      args.iterations = 5;
      args.log_each_step = 0;
    }

    global_start = MPI_Wtime();

    // Figure out how much work the average node will be doing
    args.rows_per_proc = (args.height + args.padding*2)/args.process_count;
    args.data_per_proc = args.rows_per_proc * (args.width + args.padding*2);
  }

  broadcast_and_receive_input(comm, &args);

  // Calculate the exact work each thread will do and arguments for
  // the Scatterv to scatter the grid
  int grid_size = ((args.width + args.padding*2)*(args.height + args.padding*2));
  int* data_counts = malloc(sizeof(int) * args.process_count);
  int* displacements = malloc(sizeof(int) * args.process_count);
  for (int i = 0; i < args.process_count; i++) {
    data_counts[i] = args.data_per_proc;
    displacements[i] = args.data_per_proc*sizeof(unsigned char)*i;
  }
  data_counts[args.process_count-1] += grid_size % (args.data_per_proc * args.process_count);
  unsigned char* local_data = malloc(data_counts[rank]*sizeof(unsigned char));
  memset(local_data, 0, sizeof(unsigned char) * data_counts[rank]); 

  // Scatter the data among nodes
  scatter_data(comm, &args, local_data, rank, data_counts, displacements, filename);
 
  char iteration_file[1024];

  // Local_game is our current job
  struct GAME local_game;
  local_game.grid = local_data;
  local_game.width = args.width;
  local_game.height = data_counts[rank] / (args.width + args.padding*2);
  local_game.padding = args.padding;

  // Assign halo elements to send to be received from above and below nodes
  unsigned char* halo_above = NULL;
  unsigned char* halo_below = NULL;
  if (rank > 0) {
    halo_above = (unsigned char*)malloc(sizeof(unsigned char) * (args.width + args.padding*2));
    memset(halo_above, 0, sizeof(unsigned char) * (args.width + args.padding*2));
  } 
  if (rank < args.process_count-1) {
    halo_below = (unsigned char*)malloc(sizeof(unsigned char) * (args.width + args.padding*2));
    memset(halo_below, 0, sizeof(unsigned char) * (args.width + args.padding*2));
  }

  unsigned char* global_data;
  if (rank == 0) {
    global_data = malloc(sizeof(unsigned char) * grid_size);
    memset(global_data, 0, sizeof(unsigned char) * grid_size); 
  }

  // Timing code
  double time_computing_life = 0;
  double start,end;

  for (int i = 0; i <= args.iterations; i++) {
    // Iteration 0 will just be the initial grid
    if (i > 0) {
      int total_width = args.width + args.padding*2;

      MPI_Status status;
      if (rank < args.process_count - 1) {
        MPI_Send(&local_game.grid[(local_game.height-1) * total_width], total_width, MPI_UNSIGNED_CHAR, rank+1, 1, comm); 
      }
      if (rank > 0) {
        MPI_Recv(halo_above, total_width, MPI_UNSIGNED_CHAR, rank-1, 1, comm, &status);
        MPI_Send(&local_game.grid[0], total_width, MPI_UNSIGNED_CHAR, rank-1, 0, comm); 
      }
      if (rank < args.process_count - 1) {
        MPI_Recv(halo_below, total_width, MPI_UNSIGNED_CHAR, rank+1, 0, comm, &status);
      }
      MPI_Barrier(comm);
      start = MPI_Wtime();
      // Compute the next grid 
      next(&local_game, halo_above, halo_below);
      end = MPI_Wtime();
      time_computing_life += end-start;
    }
    if (args.log_each_step) {
      // If we are logging each step, perform IO operations
      // Gather all of the local grids into global_data
      MPI_Gatherv(local_game.grid, data_counts[rank], MPI_UNSIGNED_CHAR, global_data, data_counts, displacements, MPI_UNSIGNED_CHAR, 0, comm);
      if (rank == 0) {
        #if VERBOSE == 1
          printf("\n===Iteration %i===\n", i);
          // Print the baord without the padding elements
          for (int y = args.padding; y < args.height+args.padding; y++) {
            for (int x = args.padding; x < args.width+args.padding; x++) {
              printf("%s ", global_data[y*(args.width+2*args.padding) + x] ? "X" : " ");
            }
            printf("\n");
          }
          printf("===End iteration %i===\n", i);
        #endif

        // Save to a file
        struct GAME global_game;
        global_game.grid = global_data;
        global_game.width = args.width;
        global_game.height = args.height;
        global_game.padding = args.padding;
        sprintf(iteration_file, "output/iteration-%07d.bin", i);
        write_out(iteration_file, &global_game);
      }
    }
  }

  double total_end = MPI_Wtime();
  if (rank == 0) {
    printf("\n===Timing===\nTime computing life: %f\nClock time: %f\n", time_computing_life, (total_end - global_start));
    free(local_game.grid);
    free(data_counts);
    free(halo_above);
    free(halo_below);
  } 
  MPI_Finalize();
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
