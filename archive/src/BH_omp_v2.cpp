#include "common.hpp"

#ifndef OMP_CHUNK
#define OMP_CHUNK 32
#endif

int main(int argc, char** argv) {
    std::size_t N      = 10000;
    std::size_t iters  = 10;
    int         nthreads = 1;
    if (argc > 1) N        = std::stoul(argv[1]);
    if (argc > 2) iters    = std::stoul(argv[2]);
    if (argc > 3) nthreads = std::stoi(argv[3]);

    const float dt = .5f, theta = .5f;

#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif

    std::vector<particle> tab(N);
    long t_tree=0, t_force=0, t_integrate=0;

    for (std::size_t it=0; it<iters; ++it) {

        // ── Construction de l'arbre (séquentielle) ────────────
        ticker ck;
        qtree tree(tab);
        t_tree += ck.lap();

        // ── Reset accélérations ───────────────────────────────
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<N; ++i)
            tab[i].reset_acceleration();

        // ── Calcul de forces : schedule(dynamic, chunk) ───────
        ck = ticker{};
        #pragma omp parallel for schedule(dynamic, OMP_CHUNK)
        for (std::size_t i=0; i<N; ++i)
            calc_interaction(tab[i], &tree, theta);
        t_force += ck.lap();

        // ── Intégration : schedule(static) ───────────────────
        ck = ticker{};
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<N; ++i)
            tab[i].accelerate(dt).move(dt);
        t_integrate += ck.lap();
    }

    print_timings(t_tree, t_force, t_integrate);
    return EXIT_SUCCESS;
}
