// ============================================================
//  BH_mpi_v3.cpp — MPI version 3 : arbre LOCAL + Allreduce forces
//
//  STRATÉGIE : éviter l'Allgather en construisant des arbres locaux
//  ----------------------------------------------------------------
//  Au lieu d'envoyer toutes les particules à tous les rangs
//  (Allgather O(N)), chaque rang construit son arbre sur ses seules
//  particules locales.  Les forces calculées localement sont ensuite
//  RÉDUITES (MPI_Allreduce) pour obtenir la force nette globale.
//
//  PROTOCOLE PAR ITÉRATION
//  ───────────────────────
//  1. Chaque rang r construit qtree_r sur ses particules locales.
//  2. Pour chaque particule LOCALE i du rang r :
//       force_locale[i] = calc_interaction avec qtree_r
//     (force due aux voisins locaux, approximation proche)
//  3. MPI_Allgatherv des positions SEULEMENT (3 floats / particule :
//     px, py, mass) pour construire un arbre fantôme léger.
//     → 3x moins de données que v1 (7 floats).
//  4. calc_interaction far-field sur l'arbre fantôme pour les
//     particules distantes (theta plus grand = 1.0 pour les rangs
//     distants, ce qui minimise les communications).
//  5. Les forces far-field sont ajoutées localement (pas de reduce).
//  6. accelerate + move.
//  7. Redistribution Alltoallv (même que v1).
//
//  NOTE IMPLÉMENTATION SIMPLIFIÉE
//  --------------------------------
//  Pour cette version d'exploration, on implémente une variante
//  plus simple mais instructive : Allgather positions légères
//  (3 floats : x, y, mass) pour l'arbre global + forces calculées
//  sur l'arbre léger.  Cela réduit la bande passante de 57%
//  (3 vs 7 floats par particule) au prix d'une reconstruction
//  partielle d'arbre.
//
//  AVANTAGE vs v1 et v2
//  ---------------------
//  - 57% de données MPI en moins par Allgather (3 floats vs 7).
//  - Potentiellement scalable à grand P sur cluster avec réseau lent.
//
//  INCONVÉNIENT
//  ------------
//  - L'arbre léger (positions+mass seulement) est identique à v1
//    pour le calcul BH, mais la sérialisation/désérialisation est
//    plus complexe → légère overhead CPU.
//  - La redistribution reste O(N).
// ============================================================

#include "common.hpp"

// Sérialisation "légère" : [x, y, mass] seulement
struct LightParticle {
    float x, y, mass;
};
static constexpr int LIGHT_FLOATS = 3;

// Construit un vecteur de LightParticle depuis un vecteur de particle
static void to_light(const std::vector<particle>& src, std::vector<float>& buf) {
    buf.resize(src.size() * LIGHT_FLOATS);
    for (std::size_t i=0; i<src.size(); ++i) {
        buf[i*LIGHT_FLOATS+0] = src[i].x();
        buf[i*LIGHT_FLOATS+1] = src[i].y();
        buf[i*LIGHT_FLOATS+2] = src[i].mass();
    }
}

// Reconstruit des pseudo-particles (position+masse seulement)
// pour la construction de l'arbre global far-field.
static void from_light(const std::vector<float>& buf, std::size_t n,
                        std::vector<particle>& out) {
    out.clear(); out.reserve(n);
    for (std::size_t i=0; i<n; ++i) {
        // On crée une particule à la position voulue avec la bonne masse.
        // speed/acc ne sont pas utilisés dans l'arbre (lecture seule).
        particle p(Point2D{buf[i*LIGHT_FLOATS], buf[i*LIGHT_FLOATS+1]});
        // La masse n'est pas accessible via un setter public ; on use
        // d'un trick : on l'ajoute à une particule nulle via operator+=.
        particle m(Point2D{buf[i*LIGHT_FLOATS], buf[i*LIGHT_FLOATS+1]});
        // Pour accéder à la masse on passe par la sérialisation inverse
        float tmp[particle::MPI_FLOATS] = {
            buf[i*LIGHT_FLOATS+0], buf[i*LIGHT_FLOATS+1],
            0.f, 0.f,  // speed
            buf[i*LIGHT_FLOATS+2], // mass
            0.f, 0.f   // acc
        };
        p.unpack(tmp);
        out.push_back(std::move(p));
    }
}

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

    // ── Init : broadcast depuis rang 0 ────────────────────────
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

        // ── 1. Allgather LÉGER (3 floats/particule) ───────────
        ticker ck;
        std::vector<particle> global_view;
#ifdef USE_MPI
        {
            // Construire buffer léger local
            std::vector<float> lbuf;
            to_light(local, lbuf);

            // Échanger les tailles
            int ln = (int)local.size();
            std::vector<int> all_n(mpi_size);
            MPI_Allgather(&ln,1,MPI_INT,all_n.data(),1,MPI_INT,MPI_COMM_WORLD);

            std::vector<int> counts(mpi_size), displs(mpi_size,0);
            for (int r=0;r<mpi_size;++r) counts[r]=all_n[r]*LIGHT_FLOATS;
            for (int r=1;r<mpi_size;++r) displs[r]=displs[r-1]+counts[r-1];
            const int tf = displs[mpi_size-1]+counts[mpi_size-1];

            std::vector<float> gbuf_light(tf);
            MPI_Allgatherv(lbuf.data(),(int)lbuf.size(),MPI_FLOAT,
                           gbuf_light.data(),counts.data(),displs.data(),
                           MPI_FLOAT,MPI_COMM_WORLD);

            from_light(gbuf_light, tf/LIGHT_FLOATS, global_view);
        }
#else
        global_view = local;
#endif

        // ── 2. Construction arbre global ──────────────────────
        qtree tree(global_view);
        t_tree += ck.lap();

        // ── 3. Reset + forces ──────────────────────────────────
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
