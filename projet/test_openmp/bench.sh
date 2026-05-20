#!/bin/bash

# On s'assure que le script s'arrête en cas d'erreur
set -e

echo "Compiling..."
make clean
make

echo -e "\n---> Début du Benchmark OpenMP <---\n"

# Liste des configurations de threads à tester (jusqu'aux 48 cœurs de ton nœud)
for threads in 1 2 4 8 16 32 48
do
    echo "----------------------------------------"
    echo "Running with OMP_NUM_THREADS=$threads"
    echo "----------------------------------------"
    
    # Export de la variable OpenMP et exécution
    export OMP_NUM_THREADS=$threads
    ./bench_openmp
    
    echo -e "\n"
done

echo "---> Fin du Benchmark <---"