// ============================================================
//  BH_omp_v3.cpp — OpenMP version 3 : schedule(guided)
//
//  HYPOTHÈSE TESTÉE
//  ----------------
//  Le coût par particule est hétérogène, mais on veut moins
//  d'overhead que dynamic tout en gardant un rééquilibrage.
//  schedule(guided) diminue progressivement la taille des chunks :
//  grands chunks au début (bonne localité cache), petits chunks
//  en fin de boucle (rééquilibrage fin).
//
//  JUSTIFICATION
//  -------------
//  guided chunk_min = k : les premiers chunks font N/(P) éléments,
//  puis chaque chunk suivant = (restant / P), jusqu'à un minimum k.
//  Formellement :
//    chunk_i = max(k, remaining_i / nthreads)
//
//  Avantages vs dynamic(32) :
//    1. Moins d'accès au compteur partagé (moins de chunks totaux).
//    2. Les threads travaillent sur des segments plus longs en début
//       de boucle → meilleure réutilisation du cache L1/L2 sur le
//       tableau tab[].
//    3. Les petits chunks en fin permettent quand même de rattraper
//       les stragglers.
//
//  Inconvénient :
//    Le premier thread qui commence a un gros chunk ; si ce chunk
//    est précisément la zone la plus dense (coût maximal), cela
//    peut créer un goulot.  En pratique, la distribution uniforme
//    du générateur rend cela rare.
//
//  ATTENDU
//  -------
//  Performance entre v1 (static) et v2 (dynamic) sur des nuages
//  peu hétérogènes ; potentiellement meilleure que les deux sur
//  des nuages très hétérogènes à grand N (avantage cache).
//
//  NOTE : on teste également un sort spatial préalable (tab trié
//  par coordonnée x via std::sort) qui améliore la localité cache
//  de l'arbre : les particules voisines en mémoire correspondent
//  à des régions spatiales voisines → moins de cache-miss dans
//  le parcours du qtree.  Ce sort est optionnel via -DSPATIAL_SORT.
// ============================================================

#include "common.hpp"
#include <algorithm>

#ifndef OMP_GUIDED_MIN
#define OMP_GUIDED_MIN 16
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

#ifdef SPATIAL_SORT
    // Tri spatial initial (une seule fois) : aligne l'ordre mémoire
    // sur l'ordre spatial.  Les itérations suivantes profitent de
    // la meilleure localité pour le parcours de l'arbre.
    std::sort(tab.begin(), tab.end());
#endif

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

        // ── Calcul de forces : schedule(guided, min_chunk) ────
        // Chunks décroissants : N/T → ... → OMP_GUIDED_MIN.
        // Équilibre overhead scheduling ↔ équilibrage de charge.
        ck = ticker{};
        #pragma omp parallel for schedule(guided, OMP_GUIDED_MIN)
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
