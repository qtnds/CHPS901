// ============================================================
//  BH_hybrid.cpp — Version finale : MPI × OpenMP
//
//  SYNTHÈSE DES PHASES PRÉCÉDENTES
//  ─────────────────────────────────
//  Phase OMP : dynamic(32) gagne sur static et guided pour un
//              nuage isotrope : variance de coût non nulle dans
//              calc_interaction → équilibrage dynamique payant.
//
//  Phase MPI : v1 (bandes Y + Allgather 7f) assure un équilibre
//              de charge persistant grâce à la redistribution
//              Alltoallv. v3 (allgather léger 3f) est meilleur
//              sur réseau lent ; v1 gagne sur réseau rapide.
//              On retient v1 + allgather 3f (hybridation des deux).
//
//  OPTIMISATIONS SPÉCIFIQUES À LA VERSION HYBRIDE
//  ─────────────────────────────────────────────────
//  1. FUSION reset+force : reset_acceleration() et calc_interaction()
//     sont exécutés dans le même parallel-for → une seule barrière
//     OMP, un seul pass sur le tableau local (évite cache flush).
//
//  2. ALLGATHER LÉGER (3 floats/particule) pour la construction
//     de l'arbre global : on n'envoie que (x, y, mass), soit 57 %
//     de données en moins vs 7 floats. Les vitesses et accélérations
//     ne sont pas nécessaires pour BH.
//
//  3. BUFFER MPI PERSISTANT : lbuf et gbuf alloués une seule fois
//     hors de la boucle, redimensionnés si N local change.
//
//  4. TRI SPATIAL PÉRIODIQUE : toutes les SORT_PERIOD itérations,
//     les particules locales sont triées par x → meilleure localité
//     cache lors du parcours de l'arbre (les voisins en mémoire
//     correspondent à des voisins spatiaux → moins de cache-miss).
//
//  5. COMM_US MESURÉ SÉPARÉMENT : la sortie TIMING inclut comm_us
//     pour distinguer le temps réseau du temps calcul pur.
//
//  NIVEAU THREAD MPI
//  ─────────────────
//  MPI_THREAD_FUNNELED : le thread OMP maître fait les appels MPI.
//  Les autres threads OMP ne font que du calcul local.
//
//  SORTIE
//  ──────
//  TIMING tree_us=X force_us=X integrate_us=X comm_us=X total_us=X
// ============================================================

#include "common.hpp"
#include <algorithm>

#ifndef SORT_PERIOD
#define SORT_PERIOD 10   // tri spatial toutes les 10 itérations
#endif

// ─── Sérialisation légère (x, y, mass) pour l'arbre global ───
static constexpr int LIGHT_FLOATS = 3;

static void pack_light(const std::vector<particle>& src,
                       std::vector<float>& buf) {
    buf.resize(src.size() * LIGHT_FLOATS);
    for (std::size_t i=0; i<src.size(); ++i) {
        buf[i*LIGHT_FLOATS+0] = src[i].x();
        buf[i*LIGHT_FLOATS+1] = src[i].y();
        buf[i*LIGHT_FLOATS+2] = src[i].mass();
    }
}

static void unpack_light(const std::vector<float>& buf, std::size_t n,
                         std::vector<particle>& out) {
    out.resize(n);
    for (std::size_t i=0; i<n; ++i) {
        float tmp[particle::MPI_FLOATS] = {
            buf[i*LIGHT_FLOATS+0],   // x
            buf[i*LIGHT_FLOATS+1],   // y
            0.f, 0.f,                // vx, vy (non utilisés dans l'arbre)
            buf[i*LIGHT_FLOATS+2],   // mass
            0.f, 0.f                 // ax, ay
        };
        particle p(Point2D{});
        p.unpack(tmp);
        out[i] = std::move(p);
    }
}

// ─── print_timings étendu (avec comm_us) ──────────────────────
static void print_timings_h(long tree, long force, long integrate,
                             long comm, int rank) {
    if (rank==0)
        std::cout << "TIMING"
                  << " tree_us="      << tree
                  << " force_us="     << force
                  << " integrate_us=" << integrate
                  << " comm_us="      << comm
                  << " total_us="     << (tree+force+integrate+comm)
                  << std::endl;
}

// ─────────────────────────────────────────────────────────────
int main(int argc, char** argv) {

    // ── Init MPI ──────────────────────────────────────────────
    int mpi_rank=0, mpi_size=1;
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
        std::cerr << "[WARN] rang " << mpi_rank
                  << " : MPI_THREAD_FUNNELED non garanti\n";
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

    // ── Paramètres ────────────────────────────────────────────
    std::size_t N        = 10000;
    std::size_t iters    = 10;
    int         nthreads = 1;
    if (argc > 1) N        = std::stoul(argv[1]);
    if (argc > 2) iters    = std::stoul(argv[2]);
    if (argc > 3) nthreads = std::stoi(argv[3]);

    const float dt = .5f, theta = .5f;
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif

    if (mpi_rank==0) {
        std::cout << "=== BH Hybrid MPI+OMP ===" << std::endl;
        std::cout << "N=" << N << "  iters=" << iters
                  << "  ranks=" << mpi_size
#ifdef _OPENMP
                  << "  omp_threads=" << omp_get_max_threads()
#endif
                  << "  sort_period=" << SORT_PERIOD << std::endl;
    }

    // ── Génération : rang 0 crée N particules et broadcast ────
    std::vector<float> init_buf(N * particle::MPI_FLOATS);
#ifdef USE_MPI
    if (mpi_rank==0) {
        std::vector<particle> tmp(N);
        for (std::size_t i=0; i<N; ++i) tmp[i].pack(&init_buf[i*particle::MPI_FLOATS]);
    }
    MPI_Bcast(init_buf.data(), (int)(N*particle::MPI_FLOATS),
              MPI_FLOAT, 0, MPI_COMM_WORLD);
#else
    {
        std::vector<particle> tmp(N);
        for (std::size_t i=0; i<N; ++i) tmp[i].pack(&init_buf[i*particle::MPI_FLOATS]);
    }
#endif

    // Reconstituer les particules locales (bande Y)
    std::vector<particle> all_init;
    buf_to_particles(init_buf, N, all_init);

    std::vector<particle> local;
    local.reserve(N / mpi_size + 128);

#ifdef USE_MPI
    float ymin, ymax;
    band_limits(mpi_rank, mpi_size, ymin, ymax);
    for (auto& p : all_init)
        if (p.y()>=ymin && (p.y()<ymax || mpi_rank==mpi_size-1))
            local.push_back(p);
#else
    local = std::move(all_init);
#endif

    // ── Buffers MPI persistants (Optim 3) ─────────────────────
    std::vector<float> lbuf_light;   // local sérialisé en 3f
    std::vector<float> gbuf_light;   // global reçu en 3f
    std::vector<float> lbuf_full;    // local sérialisé en 7f (redistribution)
    std::vector<int>   all_n(mpi_size, 0);
    std::vector<int>   g_counts(mpi_size, 0);
    std::vector<int>   g_displs(mpi_size, 0);

    long t_tree=0, t_force=0, t_integrate=0, t_comm=0;

    for (std::size_t it=0; it<iters; ++it) {

        const std::size_t nl = local.size();

        // ── Optim 4 : tri spatial périodique ──────────────────
        if (it>0 && it % SORT_PERIOD == 0)
            std::sort(local.begin(), local.end());

        // ── Communication : Allgather léger (Optim 2) ─────────
        ticker ck_comm;
#ifdef USE_MPI
        {
            pack_light(local, lbuf_light);
            int ln = (int)local.size();
            MPI_Allgather(&ln, 1, MPI_INT,
                          all_n.data(), 1, MPI_INT, MPI_COMM_WORLD);

            int total_floats = 0;
            for (int r=0; r<mpi_size; ++r) {
                g_counts[r] = all_n[r] * LIGHT_FLOATS;
                g_displs[r] = total_floats;
                total_floats += g_counts[r];
            }
            gbuf_light.resize(total_floats);

            MPI_Allgatherv(lbuf_light.data(), (int)lbuf_light.size(), MPI_FLOAT,
                           gbuf_light.data(), g_counts.data(), g_displs.data(),
                           MPI_FLOAT, MPI_COMM_WORLD);
        }
        t_comm += ck_comm.lap();

        // Désérialiser la vue globale (positions+masses uniquement)
        std::vector<particle> global_view;
        unpack_light(gbuf_light, gbuf_light.size()/LIGHT_FLOATS, global_view);
#else
        const std::vector<particle>& global_view = local;
#endif

        // ── Construction de l'arbre global (séquentielle) ─────
        ticker ck;
        qtree tree(global_view);
        t_tree += ck.lap();

        // ── Forces : reset + interaction fusionnés (Optim 1) ──
        // Un seul parallel-for, une seule barrière OMP.
        // schedule(dynamic,32) : chunk de 32 = compromis empirique
        // entre overhead scheduling et équilibrage de charge.
        ck = ticker{};
        #pragma omp parallel for schedule(dynamic, 32)
        for (std::size_t i=0; i<nl; ++i) {
            local[i].reset_acceleration();
            calc_interaction(local[i], &tree, theta);
        }
        t_force += ck.lap();

        // ── Intégration ───────────────────────────────────────
        // schedule(static) : coût uniforme, overhead nul.
        ck = ticker{};
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<nl; ++i)
            local[i].accelerate(dt).move(dt);
        t_integrate += ck.lap();

        // ── Redistribution : Alltoallv des particules migrantes
#ifdef USE_MPI
        ck_comm = ticker{};
        redistribute_particles(local, mpi_rank, mpi_size);
        MPI_Barrier(MPI_COMM_WORLD);
        t_comm += ck_comm.lap();
#endif
    }

    // ── Timings ───────────────────────────────────────────────
    print_timings_h(t_tree, t_force, t_integrate, t_comm, mpi_rank);

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
}
