# Barnes-Hut N-body — Étude de parallélisation OMP + MPI

## Vue d'ensemble

Ce projet explore la parallélisation du simulateur Barnes-Hut en deux phases
successives : on choisit d'abord la meilleure stratégie OpenMP, puis on l'intègre
dans trois architectures MPI différentes pour les comparer.

```
Phase 1 — OpenMP (4 variantes)
    bh_omp_v1   schedule(static)       sur la boucle de forces
    bh_omp_v2   schedule(dynamic, 32)  sur la boucle de forces
    bh_omp_v3   schedule(guided, 16)   sur la boucle de forces
    bh_omp_v3s  guided + tri spatial   (amélioration cache)
        ↓ meilleure version choisie par benchmark
Phase 2 — MPI × OMP (3 architectures)
    bh_mpi_v1   bandes Y   + Allgather 7 floats/particule
    bh_mpi_v2   blocs idx  + Allgather 7 floats/particule (pas de redist.)
    bh_mpi_v3   bandes Y   + Allgather 3 floats/particule (comm. légère)
```

---

## Structure

```
bh_parallel/
├── src/
│   ├── common.hpp      Types partagés (Point2D, particle, qtree, MPI helpers)
│   ├── BH_omp_v1.cpp   OMP — schedule(static)
│   ├── BH_omp_v2.cpp   OMP — schedule(dynamic, 32)
│   ├── BH_omp_v3.cpp   OMP — schedule(guided, 16)  + option SPATIAL_SORT
│   ├── BH_mpi_v1.cpp   MPI — bandes Y + Allgather complet
│   ├── BH_mpi_v2.cpp   MPI — blocs index + Allgather complet
│   └── BH_mpi_v3.cpp   MPI — bandes Y + Allgather léger (3 floats)
├── scripts/
│   ├── benchmark.sh    Benchmark complet (phases OMP puis MPI)
│   ├── analyse.py      Tableaux + figures PNG
│   └── run_cluster.sh  Script SLURM pour cluster
├── bin/                Binaires (créé par make)
├── results/            CSV + figures (créé par benchmark)
└── Makefile
```

---

## Installation

### Local (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y g++ make libopenmpi-dev openmpi-bin python3-pip
pip3 install matplotlib numpy
```

### Vérification

```bash
g++ --version          # ≥ 9
mpicxx --version
echo '#include <omp.h>
int main(){}' | g++ -fopenmp -x c++ - -o /dev/null && echo "OMP OK"
```

---

## Compilation

```bash
make all          # compile les 8 binaires
make seq          # séquentiel uniquement
make omp          # 4 variantes OMP
make mpi          # 3 variantes MPI
make clean
```

---

## Exécution manuelle

```
./bin/<binaire>  [N_particules]  [N_iters]  [N_threads_OMP]
```

```bash
# Séquentiel
./bin/bh_seq 50000 10

# OMP v2 — 2 threads
./bin/bh_omp_v2 50000 10 2

# MPI v1 — 2 rangs × 2 threads OMP
mpirun -n 2 -x OMP_NUM_THREADS=2 ./bin/bh_mpi_v1 50000 10 2

# Sur 2 cœurs physiques seulement (--oversubscribe pour le dev)
mpirun --oversubscribe -n 4 ./bin/bh_mpi_v1 10000 5 1
```

---

## Benchmark

```bash
make benchmark        # complet (~20 min selon la machine)
make benchmark_quick  # rapide  (~2 min, N réduit)
```

Résultats dans `results/` :

| Fichier | Contenu |
|--------|---------|
| `seq_results.csv` | Référence séquentielle |
| `omp_results.csv` | Toutes variantes OMP par (N, T) |
| `mpi_results.csv` | Toutes variantes MPI par (N, R, T) |

### Analyse et figures

```bash
python3 scripts/analyse.py
# ou
python3 scripts/analyse.py --results results/
```

Figures générées dans `results/figures/` :

| Figure | Description |
|-------|-------------|
| `fig1_omp_speedup_vs_N.png`  | Speedup OMP selon N (T=max) |
| `fig2_omp_speedup_vs_T.png`  | Speedup OMP selon nb threads (N=max) |
| `fig3_omp_efficiency.png`    | Efficacité parallèle OMP |
| `fig4_mpi_speedup_vs_ranks.png` | Speedup MPI selon nb rangs |
| `fig5_hybrid_speedup.png`    | Speedup hybride (rangs × threads) |
| `fig6_omp_time_breakdown.png` | Décomposition tree/force/integ OMP |
| `fig7_mpi_time_breakdown.png` | Décomposition tree/force/integ MPI |

---

## Stratégie de parallélisation

### Ce qui n'est PAS parallélisé : le qtree

La construction du qtree est séquentielle dans toutes les versions.
Les insertions récursives modifient des nœuds partagés → rendre cela
thread-safe nécessiterait des verrous qui annuleraient le gain.
Le coût est O(N log N) soit ≈ 5–15 % du total.

### Phase 1 — OpenMP : trois stratégies sur la boucle de forces

La boucle de forces est le goulot d'étranglement (80–90 % du temps).
Le coût de `calc_interaction(p, tree, θ)` par particule est **variable** :
il dépend de la position relative dans l'arbre.

#### v1 — `schedule(static)`

Découpe le tableau en `N/T` blocs contigus fixes. Overhead zéro.
Hypothèse : la variance du coût est faible (distribution isotrope).
Avantage : meilleure localité cache (blocs contigus).

#### v2 — `schedule(dynamic, 32)`

Chaque thread demande un chunk de 32 particules dès qu'il est libre.
Rééquilibre dynamiquement la charge. Overhead : accès concurrent à
un compteur atomique. Recommandé quand la distribution est hétérogène.
**C'est cette version qui est intégrée dans les binaires MPI.**

#### v3 — `schedule(guided, 16)`

Chunks décroissants : `N/T` → ... → 16. Compromis entre static
(bonne localité) et dynamic (équilibrage). Variante v3s : tri spatial
préalable du tableau pour aligner l'ordre mémoire sur l'espace.

### Phase 2 — MPI : trois architectures

#### v1 — Bandes Y + Allgather complet (7 floats)

Décomposition géométrique classique. Simple.  
Communication : `Allgatherv(7N floats)` + `Alltoallv(redistribution)`.

#### v2 — Blocs d'index + Allgather complet (pas de redistribution)

Affectation statique par tranche d'indices. Pas de migration → pas
d'Alltoallv. Un seul Allgatherv par itération mais on reconstruit l'arbre
sur `all_p` complet. Peut souffrir de déséquilibre de charge si les clusters
gravitationnels se forment (zones denses sur certains rangs).

#### v3 — Bandes Y + Allgather léger (3 floats : x, y, mass)

Réduit les données MPI de 57 % (3 vs 7 floats/particule) en n'échangeant
que les informations nécessaires à la construction de l'arbre BH. Plus
efficace sur réseaux lents (ethernet), négligeable sur InfiniBand.

### Niveau thread MPI

Toutes les versions utilisent `MPI_THREAD_FUNNELED` : seul le thread
maître OpenMP appelle MPI. Compatible avec toutes les implémentations.

---

## Portage cluster SLURM

```bash
# Copier le projet
scp -r bh_parallel/ user@cluster:~/

# Sur le nœud de login
cd ~/bh_parallel
# Adapter les modules dans scripts/run_cluster.sh
# module load gcc/12 openmpi/4.1
make all

# Soumettre
sbatch scripts/run_cluster.sh

# Suivre
squeue -u $USER
tail -f results/slurm_<JOBID>.out
```

Config recommandée sur cluster :

| N particules | Rangs | Threads/rang | Commentaire |
|-------------|-------|--------------|-------------|
| ≤ 10 000    | 1     | = nb cœurs   | OMP pur suffit |
| ≤ 100 000   | 4–8   | 4–8          | Hybride recommandé |
| > 100 000   | 8–16  | 4–8          | v3 MPI (comm. légère) |
