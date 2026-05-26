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

    // ── Répartition statique par index ────────────────────────
    // Rang r : particules [r_start .. r_end)
    const std::size_t chunk = (N + mpi_size - 1) / mpi_size;
    const std::size_t r_start = (std::size_t)mpi_rank * chunk;
    const std::size_t r_end   = std::min(r_start + chunk, N);
    const std::size_t nl      = r_end - r_start;

    // Rang 0 génère TOUTES les particules, broadcast
    std::vector<particle> all_p;
    std::vector<float> gbuf;
#ifdef USE_MPI
    if (mpi_rank==0) { all_p.resize(N); particles_to_buf(all_p, gbuf); }
    else gbuf.resize(N*particle::MPI_FLOATS);
    MPI_Bcast(gbuf.data(),(int)(N*particle::MPI_FLOATS),MPI_FLOAT,0,MPI_COMM_WORLD);
    if (mpi_rank!=0) buf_to_particles(gbuf, N, all_p);
#else
    all_p.resize(N);
#endif

    // Chaque rang extrait sa tranche
    std::vector<particle> local(all_p.begin()+r_start, all_p.begin()+r_end);

    long t_tree=0, t_force=0, t_integrate=0;

    for (std::size_t it=0; it<iters; ++it) {

        // ── 1. Allgather → vue globale à jour ─────────────────
#ifdef USE_MPI
        // Sérialiser local → buf temporaire, puis Allgatherv
        std::vector<float> lbuf;
        particles_to_buf(local, lbuf);
        // counts et displs fixes (blocs statiques)
        std::vector<int> counts(mpi_size), displs(mpi_size, 0);
        for (int r=0; r<mpi_size; ++r) {
            const std::size_t rs = (std::size_t)r * chunk;
            const std::size_t re = std::min(rs+chunk, N);
            counts[r] = (int)(re-rs) * particle::MPI_FLOATS;
        }
        for (int r=1; r<mpi_size; ++r) displs[r]=displs[r-1]+counts[r-1];
        const int tf = displs[mpi_size-1]+counts[mpi_size-1];
        std::vector<float> allbuf(tf);
        MPI_Allgatherv(lbuf.data(),(int)lbuf.size(),MPI_FLOAT,
                       allbuf.data(),counts.data(),displs.data(),MPI_FLOAT,MPI_COMM_WORLD);
        buf_to_particles(allbuf, N, all_p);
#else
        (void)nl;
#endif

        // ── 2. Construction arbre global ──────────────────────
        ticker ck;
        qtree tree(all_p);
        t_tree += ck.lap();

        // ── 3. Reset + forces sur la tranche locale ────────────
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

#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    print_timings(t_tree, t_force, t_integrate, mpi_rank);
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
}
