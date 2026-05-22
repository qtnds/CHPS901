// ============================================================
//  BH_mpi_v1.cpp — MPI version 1 : bandes horizontales + Allgather
//
//  STRATÉGIE : décomposition géométrique par bandes Y
//  ---------------------------------------------------
//  L'espace [-0.5, 0.5]² est découpé en P bandes horizontales
//  égales (une par rang).  Chaque rang "possède" les particules
//  dont la coordonnée y est dans sa bande.
//
//  PROTOCOLE PAR ITÉRATION
//  ───────────────────────
//  1. MPI_Allgatherv  : chaque rang envoie ses particules locales
//     à tous les autres.  Après cet appel, chaque rang dispose
//     d'une copie complète des N particules → construction du
//     qtree GLOBAL identique sur chaque rang.
//  2. Construction qtree global (séquentielle sur chaque rang).
//  3. calc_interaction sur les particules LOCALES seulement
//     (→ schedule(dynamic,32) OMP).
//  4. accelerate + move (local).
//  5. MPI_Alltoallv : redistribuer les particules qui ont migré
//     hors de la bande du rang courant.
//
//  AVANTAGE
//  --------
//  Simple à implémenter.  La bande Y est une décomposition
//  naturelle pour les particules uniformément distribuées.
//
//  INCONVÉNIENT
//  ------------
//  L'Allgather envoie O(N × 7 floats) à chaque itération.
//  Pour N = 100 000 et 4 rangs, cela représente ~2.8 Mo par
//  itération par rang.  Sur un réseau lent (ethernet 1 Gb/s),
//  c'est ~22 ms par itération → peut dominer le calcul.
//  Sur InfiniBand ou dans un seul nœud (shared memory), c'est
//  négligeable.
//
//  NIVEAU THREAD MPI
//  -----------------
//  MPI_THREAD_FUNNELED : seul le thread maître OMP appelle MPI.
//  Les threads OMP ne font que du calcul pur sur les particules
//  locales.  Compatible avec la plupart des implémentations MPI.
// ============================================================

#include "common.hpp"

int main(int argc, char** argv) {
    int mpi_rank=0, mpi_size=1;
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

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

    // ── Initialisation : rang 0 génère, broadcast ─────────────
    std::vector<particle> all_p;
    std::vector<float> gbuf;
#ifdef USE_MPI
    if (mpi_rank==0) { all_p.resize(N); particles_to_buf(all_p, gbuf); }
    else gbuf.resize(N*particle::MPI_FLOATS);
    MPI_Bcast(gbuf.data(),(int)(N*particle::MPI_FLOATS),MPI_FLOAT,0,MPI_COMM_WORLD);
    if (mpi_rank!=0) buf_to_particles(gbuf, N, all_p);

    float ymin, ymax;
    band_limits(mpi_rank, mpi_size, ymin, ymax);
    std::vector<particle> local;
    for (auto& p : all_p)
        if (p.y()>=ymin && (p.y()<ymax || mpi_rank==mpi_size-1))
            local.push_back(p);
#else
    all_p.resize(N);
    std::vector<particle>& local = all_p;
#endif

    long t_tree=0, t_force=0, t_integrate=0;

    for (std::size_t it=0; it<iters; ++it) {

        // ── 1. Allgather → vue globale ─────────────────────────
        std::vector<particle> global_view;
#ifdef USE_MPI
        allgather_particles(local, global_view, mpi_size);
#else
        global_view = local;
#endif

        // ── 2. Construction arbre global (séquentielle) ────────
        ticker ck;
        qtree tree(global_view);
        t_tree += ck.lap();

        // ── 3. Reset + forces (OMP dynamic sur particules locales)
        const std::size_t nl = local.size();
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<nl; ++i) local[i].reset_acceleration();

        ck = ticker{};
        #pragma omp parallel for schedule(dynamic, 32)
        for (std::size_t i=0; i<nl; ++i) calc_interaction(local[i], &tree, theta);
        t_force += ck.lap();

        // ── 4. Intégration ─────────────────────────────────────
        ck = ticker{};
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<nl; ++i) local[i].accelerate(dt).move(dt);
        t_integrate += ck.lap();

        // ── 5. Redistribution ──────────────────────────────────
#ifdef USE_MPI
        redistribute_particles(local, mpi_rank, mpi_size);
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    print_timings(t_tree, t_force, t_integrate, mpi_rank);
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
}
