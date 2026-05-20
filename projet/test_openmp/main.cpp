#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

int main() {
    int max_threads = omp_get_max_threads();
    std::cout << "========================================" << std::endl;
    std::cout << "Nombre max de threads OpenMP dispo : " << max_threads << std::endl;
    std::cout << "========================================" << std::endl;

    // On augmente un peu la taille pour avoir une mesure stable
    const size_t N = 200000000; 
    double sum = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    // L'astuce : on utilise 'i' pour que le calcul soit unique à chaque itération
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; ++i) {
        double val = static_cast<double>(i) * 0.00001;
        sum += std::sin(val) * std::cos(val);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Resultat de la somme : " << sum << std::endl;
    std::cout << "Temps d'execution    : " << duration.count() << " secondes" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}