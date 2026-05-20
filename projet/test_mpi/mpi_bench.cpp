#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main(int argc, char** argv) {
    // Initialisation de l'environnement MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "Lancement du Benchmark avec " << size << " processus MPI" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Nombre total de points à générer par processus
    const long long local_samples = 50000000; // 50 millions par processus
    long long local_inside = 0;

    // Initialisation d'un générateur de nombres aléatoires unique par processus
    std::mt19937 prng(1337 + rank);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Début du chrono
    auto start = std::chrono::high_resolution_clock::now();

    // Méthode de Monte-Carlo : on compte les points dans le quart de cercle
    for (long long i = 0; i < local_samples; ++i) {
        double x = dist(prng);
        double y = dist(prng);
        if (x * x + y * y <= 1.0) {
            local_inside++;
        }
    }

    // Réduction : on somme les résultats de tous les processus sur le rang 0
    // Réduction : on somme les résultats de tous les processus sur le rang 0
    long long global_inside = 0;
    MPI_Reduce(&local_inside, &global_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Le rang 0 calcule le temps max et affiche le résultat
    double local_time = duration.count();
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long total_samples = local_samples * size;
        double pi_estimate = 4.0 * static_cast<double>(global_inside) / total_samples;
        
        std::cout << "Points calculés au total : " << total_samples << std::endl;
        std::cout << "Estimation de Pi         : " << pi_estimate << std::endl;
        std::cout << "Temps de calcul (max)    : " << max_time << " secondes" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}