#!/bin/bash
#SBATCH --time=0:10:00 # walltime, abbreviated by -t
#SBATCH --nodes=2     # number of cluster nodes, abbreviated by -N
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using job and first node values
#SBATCH --ntasks=24   # number of MPI tasks, abbreviated by -n
# additional information for allocated clusters
#SBATCH --account=usucs5030     # account - abbreviated by -A
#SBATCH --partition=kingspeak  # partition, abbreviated by -p

cd $HOME/gol/mpi
mkdir timing-study

module load intel mpich 

iterations=1000

for cores in 1 4 8 12 16 20 #24
do
  for size in 250 500 750 1000 1250 1500 1750 2000
  do
    mpirun -np $cores ./gol simulate random $size $size $iterations 1 > timing-study/output-$cores-$iterations-$size.txt
  done
done
