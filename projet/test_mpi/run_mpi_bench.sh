#!/bin/bash
set -e

echo "Compilation du benchmark MPI..."
make clean
make

echo -e "\n---> Début du Benchmark MPI (via mpirun) <---\n"

# Liste des processus MPI à tester sur ton allocation
for ranks in 1 2 4 8 16 32 48
do
    echo "----------------------------------------"
    echo "Running with $ranks MPI Process"
    echo "----------------------------------------"
    
    # -np spécifie le nombre de processus (ranks)
    # --oversubscribe permet de forcer l'exécution si le binding de ton bash est trop strict
    mpirun -np $ranks --oversubscribe ./bench_mpi
    
    echo -e "\n"
done

echo "---> Fin du Benchmark MPI <---"