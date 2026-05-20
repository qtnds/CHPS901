// ============================================================
//  BH_omp_v2.cpp — OpenMP version 2 : schedule(dynamic, chunk)
//
//  HYPOTHÈSE TESTÉE
//  ----------------
//  Le coût par particule dans calc_interaction est NON UNIFORME.
//  On utilise un ordonnancement dynamique : chaque thread demande
//  un nouveau chunk de travail dès qu'il a terminé le précédent.
//  Cela équilibre la charge au prix d'un overhead de scheduling
//  (accès concurrent à un compteur atomique partagé).
//
//  JUSTIFICATION
//  -------------
//  Dans BH, la profondeur de récursion de calc_interaction dépend
//  de la distance particule↔nœud.  Pour un nuage avec gradient de
//  densité (bords vs centre), les particules centrales font plus
//  de travail.  static crée alors un "stragglers problem" : les
//  threads traitant les zones denses finissent en retard, pendant
//  que les autres attendent à la barrière implicite de la boucle.
//
//  CHOIX DU CHUNK
//  --------------
//  chunk = 32 : empiriquement bon pour N ~ 10k-100k.
//    - trop petit (1) : overhead de scheduling dominant.
//    - trop grand (N/T) : dégénère en static, perd l'équilibrage.
//  On mesure aussi un chunk = 64 via la version v2b (même fichier,
//  voir Makefile) pour voir la sensibilité.
//
//  ATTENDU
//  -------
//  Meilleur que v1 quand la distribution est hétérogène ;
//  légèrement moins bon quand elle est uniforme (overhead dy.).
// ============================================================

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
        // Le thread qui termine son chunk en demande un nouveau
        // immédiatement → pas d'attente aux barrières internes.
        // OMP_CHUNK est défini à la compilation (défaut 32).
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
