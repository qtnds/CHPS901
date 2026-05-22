// ============================================================
//  BH_omp_v1.cpp — OpenMP version 1 : schedule(static) partout
//
//  HYPOTHÈSE TESTÉE
//  ----------------
//  La boucle de calcul de forces est traitée comme si le coût
//  par particule était UNIFORME : on découpe le tableau en blocs
//  continus de taille égale (schedule static, chunk automatique
//  = N/T).  C'est la forme la plus simple et la plus légère en
//  overhead de scheduling.
//
//  JUSTIFICATION
//  -------------
//  En théorie, BH est NON uniforme : une particule proche du
//  centre de masse d'un nœud interne déclenche une descente
//  récursive profonde, une particule loin s'en sort avec une
//  seule approximation.  Mais pour un nuage ISOTROPE (notre
//  générateur uniforme), la variance du coût par particule est
//  modérée.  L'idée est de mesurer si l'overhead zéro de static
//  compense le léger déséquilibre de charge.
//
//  ATTENDU
//  -------
//  - Speedup ≈ T sur machine à faible hétérogénéité spatiale.
//  - Peut être moins bon que dynamic si les particules ont des
//    densités locales très variables (clusters denses vs vide).
//
//  CE QUI N'EST PAS PARALLÉLISÉ
//  -----------------------------
//  La construction du qtree reste séquentielle : les insertions
//  modifient récursivement des nœuds partagés → impossible sans
//  verrous coûteux. Son coût est O(N log N) soit ~5-15% du total.
// ============================================================

#include "common.hpp"

int main(int argc, char** argv) {
    std::size_t N      = 10000;
    std::size_t iters  = 10;
    int         nthreads = 1; (void)nthreads;
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
        // schedule(static) : coût uniforme, zéro overhead.
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<N; ++i)
            tab[i].reset_acceleration();

        // ── Calcul de forces : schedule(static) ───────────────
        // Chunk automatique = N/nthreads blocs contigus.
        // Avantage  : overhead minimal, bonne localité cache
        //             (chaque thread lit un segment contigu).
        // Inconvénient : si le coût varie (zones denses vs vides)
        //                certains threads terminent avant les autres.
        ck = ticker{};
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<N; ++i)
            calc_interaction(tab[i], &tree, theta);
        t_force += ck.lap();

        // ── Intégration : schedule(static) ───────────────────
        // Coût parfaitement uniforme → static est optimal.
        ck = ticker{};
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<N; ++i)
            tab[i].accelerate(dt).move(dt);
        t_integrate += ck.lap();
    }

    print_timings(t_tree, t_force, t_integrate);
    return EXIT_SUCCESS;
}
