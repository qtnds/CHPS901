# Barnes-Hut N-body — Parallélisation OpenMP + MPI + CUDA

Étude comparative de parallélisation d'un simulateur gravitationnel 2D
basé sur l'algorithme de Barnes-Hut, sur cluster HPC Romeo (NVIDIA GH200).

```
Référence    bh_seq       BH_omp_v1 sans OMP — baseline pour les speedups
Phase 1      OMP          4 stratégies de scheduling sur la boucle de forces
Phase 2      MPI × OMP    3 architectures de communication
Phase 3      Hybride      Version finale MPI+OMP optimisée
Phase 4      CUDA         Kernel GPU avec pipeline CPU→sérialisation→GPU
```

---

## Structure du projet

```
projet4/
├── src/
│   ├── common.hpp        Types partagés (particle, qtree, helpers MPI/timer)
│   ├── BH_omp_v1.cpp     OMP — schedule(static)
│   ├── BH_omp_v2.cpp     OMP — schedule(dynamic, 32)
│   ├── BH_omp_v3.cpp     OMP — schedule(guided, 16)  [-DSPATIAL_SORT pour v3s]
│   ├── BH_mpi_v1.cpp     MPI — bandes Y + Allgather 7f + Alltoallv
│   ├── BH_mpi_v2.cpp     MPI — blocs idx + Allgather 7f (sans redistrib.)
│   ├── BH_mpi_v3.cpp     MPI — bandes Y + Allgather 3f (comm. légère)
│   ├── BH_hybrid.cpp     Hybride MPI+OMP final
│   └── BH_cuda.cu        Pipeline CPU+GPU (arbre CPU → SOA → kernel forces)
├── scripts/
│   ├── benchmark.sh      Orchestration des 4 phases → CSVs
│   ├── analyse.py        Tableaux console + 13 figures PNG
│   └── run_cluster.sh    Script SBATCH pour soumettre sans nœud interactif
├── bin/                  Binaires (créé par make)
├── results/              CSVs + figures (créé par benchmark)
├── requirements.txt      Dépendances Python (matplotlib, numpy)
└── Makefile
```

---

## Mise en place

### 1. Cloner le dépôt

```bash
git clone <url-du-depot> projet4
cd projet4
```

### 2. Environnement Python (pour analyse.py)

```bash
# Créer le venv
python -m venv .venv

# Activer le venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

> **Note :** le venv n'est nécessaire que pour `python3 scripts/analyse.py`.
> La compilation et le benchmark ne dépendent pas de Python.

---

## Utilisation sur Romeo

Il y a deux modes : tester interactivement sur un nœud alloué, ou soumettre
directement en batch avec `sbatch`.

---

### Mode interactif (tests par phases, mise au point)

Ce mode alloue un nœud pour la durée de la session. Utile pour tester
chaque version indépendamment et voir les résultats immédiatement.

#### Étape 1 — Allouer un nœud

```bash
salloc -t 4:00:00 --account=r250123 --constraint="armgpu" \
       --cores=48 -N 4 --mem=10G --gpus=4 --exclusive
```

#### Étape 2 — Ouvrir un shell sur le nœud alloué

```bash
srun --pty bash
```

#### Étape 3 — Charger l'environnement

```bash
romeo_load_armgpu_env
spack load gcc@11.4.1   /nrzc6ai
spack load openmpi@4.1.7 /nkokjyt
spack load cuda@12.6.2  /3mzltpz
```

#### Étape 4 — Compiler

```bash
make all
```

#### Étape 5 — Tester les versions une par une

```bash
# ── Séquentiel (référence)
./bin/bh_seq 10000 10 1
# → TIMING tree_us=X force_us=X integrate_us=X total_us=X

# ── OpenMP — 4 variantes
./bin/bh_omp_v1 10000 10 4     # static,    4 threads
./bin/bh_omp_v2 10000 10 4     # dynamic32, 4 threads
./bin/bh_omp_v3 10000 10 4     # guided16,  4 threads
./bin/bh_omp_v3s 10000 10 4    # guided16 + tri spatial

# ── MPI — 3 variantes (2 rangs × 4 threads)
export OMP_NUM_THREADS=4
mpirun --oversubscribe -np 2 ./bin/bh_mpi_v1 10000 10 4
mpirun --oversubscribe -np 2 ./bin/bh_mpi_v2 10000 10 4
mpirun --oversubscribe -np 2 ./bin/bh_mpi_v3 10000 10 4

# ── Hybride (2 rangs × 4 threads)
mpirun --oversubscribe -np 2 ./bin/bh_hybrid 10000 10 4

# ── CUDA (GPU GH200)
./bin/bh_cuda 10000 10 1
# → GPU=NVIDIA GH200 120GB  SM=...  Mem=...MB
# → TIMING tree_us=X serial_us=X gpu_us=X integrate_us=X total_us=X
```

#### Étape 6 — Benchmark complet depuis le nœud interactif

```bash
# Benchmark rapide (~15 min)
bash scripts/benchmark.sh --quick

# Benchmark complet (~2-3h)
bash scripts/benchmark.sh

# Une seule phase
bash scripts/benchmark.sh --phase 4   # CUDA uniquement
```

---

### Mode batch SBATCH (sans réservation interactive)

Soumettre depuis le **nœud de login**, le job s'exécutera dès qu'un
nœud est disponible.

```bash
# Benchmark complet
sbatch scripts/run_cluster.sh

# Benchmark rapide
sbatch scripts/run_cluster.sh --quick

# Phase CUDA seule (adapter la partition dans run_cluster.sh)
sbatch scripts/run_cluster.sh --phase 4
```

**Suivi du job :**

```bash
squeue -u $USER                           # statut du job
tail -f results/slurm_<JOBID>.out         # logs en temps réel
scancel <JOBID>                           # annuler si besoin
```

> Les directives `#SBATCH` (partition, ntasks, cpus-per-task, time) se
> trouvent en haut de `scripts/run_cluster.sh`. Adapter si nécessaire.

---

## Ce que mesure le benchmark

`benchmark.sh` orchestre 4 phases en mesurant la **médiane de 3 exécutions**
pour chaque configuration. Chaque binaire affiche une ligne `TIMING` parsée
automatiquement.

### Format de sortie TIMING

```
TIMING tree_us=X force_us=X integrate_us=X total_us=X
```

Les variantes MPI/hybride ajoutent `comm_us`, la version CUDA ajoute
`serial_us` et `gpu_us`.

### Phases et paramètres mesurés

| Phase | Binaires | Paramètres variés | CSV produit |
|-------|----------|-------------------|-------------|
| Séquentiel | `bh_seq` | N | `seq_results.csv` |
| Phase 1 OMP | `bh_omp_v1..v3s` | N × T (threads) | `omp_results.csv` |
| Phase 2 MPI | `bh_mpi_v1..v3` | N × R (rangs) × T | `mpi_results.csv` |
| Phase 3 Hybride | `bh_hybrid` | N × R × T | `hybrid_results.csv` |
| Phase 4 CUDA | `bh_cuda` | N étendu × T | `cuda_results.csv` |

### Champs mesurés par phase

**Séquentiel & OMP** — décomposition en 3 phases :
- `tree_us` : construction du quadtree (séquentielle, non parallélisée)
- `force_us` : calcul des forces gravitationnelles (parallélisé OMP)
- `integrate_us` : intégration Euler (parallélisée OMP)

**MPI & Hybride** — ajout de :
- `comm_us` : temps total des communications MPI par itération
  (Allgatherv + Alltoallv le cas échéant)

**CUDA** — décomposition en 4 phases :
- `tree_us` : construction du quadtree (CPU, séquentielle)
- `serial_us` : sérialisation de l'arbre en buffer AOS + transfert H2D
- `gpu_us` : kernel de forces sur GPU + transfert D2H des accélérations
- `integrate_us` : intégration Euler (CPU)

### Métriques calculées

Pour chaque mesure, le benchmark calcule :
- **Speedup** : `T_seq / T_parallèle` — combien de fois plus rapide que `bh_seq`
- **Efficacité** : `speedup / nb_procs` — utilisation effective des ressources

### Plage CUDA étendue

La phase CUDA teste une plage de N plus large que les autres phases
(`PARTICLES_CUDA=1000 5000 10000 50000 100000 500000`) pour identifier le
**point de crossover** : la valeur de N à partir de laquelle le GPU devient
plus rapide que le CPU. En dessous de ce seuil, l'overhead de sérialisation
et de transfert H2D domine.

---

## Ce qu'affiche analyse.py

```bash
source .venv/bin/activate
python3 scripts/analyse.py               # lit results/ par défaut
python3 scripts/analyse.py --results results/
```

### Sortie console

Tableaux de résultats pour chaque phase, avec pour chaque (N, T, R) :
- temps total en µs/ms/s selon l'ordre de grandeur
- décomposition tree / force / intégration / comm en µs
- speedup calculé vs référence séquentielle

### 13 figures PNG dans `results/figures/`

| Figure | Description |
|--------|-------------|
| `fig01_ref_breakdown.png` | Barres empilées tree/force/intég de `bh_seq` vs N — montre que **85–94 %** du temps est dans le calcul de forces |
| `fig02_omp_speedup_vs_N.png` | Speedup des 4 variantes OMP en fonction de N (T=max) |
| `fig03_omp_speedup_vs_T.png` | Speedup OMP en fonction du nombre de threads (N=max) — comparé à l'idéal linéaire |
| `fig04_omp_efficiency.png` | Efficacité parallèle OMP en % (100 % = idéal) |
| `fig05_omp_breakdown.png` | Barres empilées tree/force/intég pour chaque variante OMP (N=max, T=max) |
| `fig06_mpi_speedup_vs_ranks.png` | Speedup des 3 variantes MPI en fonction du nombre de rangs |
| `fig07_mpi_breakdown.png` | Barres empilées tree/force/intég pour chaque variante MPI (N=max, R=max) |
| `fig08_hybrid_speedup.png` | Speedup hybride vs nombre de processus totaux (R×T) — avec courbe idéale |
| `fig09_hybrid_breakdown.png` | Barres empilées avec `comm_us` pour chaque N — montre le poids des communications |
| `fig10_cuda_breakdown_vs_N.png` | Barres empilées tree/serial/GPU/intég vs N — montre comment les parts évoluent |
| `fig11_cuda_crossover.png` | Courbes temps absolu CUDA vs séquentiel vs hybride — ligne verticale au **point de crossover** |
| `fig12_cuda_breakdown_pct.png` | Stacked area en % : à petit N la sérialisation domine, à grand N le kernel GPU prend le dessus |
| `fig13_global_comparison.png` | Vue synthétique : speedup de la **meilleure config** de chaque phase sur le même graphe |

---

## Compilation seule

```bash
make all              # tous les binaires
make seq              # bh_seq uniquement
make omp              # 4 variantes OMP
make mpi              # 3 variantes MPI
make hybrid           # bh_hybrid
make cuda             # bh_cuda GPU réel (nvcc requis, CUDA_ARCH=sm_90)
make cuda_cpu         # bh_cuda fallback CPU (sans nvcc)
make clean            # supprime bin/ et results/
```

Pour changer l'architecture CUDA (si autre que GH200) :

```bash
make cuda CUDA_ARCH=sm_80   # A100
make cuda CUDA_ARCH=sm_86   # RTX 3090
make cuda CUDA_ARCH=sm_70   # V100
```

---

## Paramétrage du benchmark

Les paramètres par défaut sont surchargables via variables d'environnement :

```bash
export BH_PARTICLES="10000 50000 100000 500000"
export BH_PARTICLES_CUDA="1000 10000 100000 500000"
export BH_ITERS=20
export BH_OMP_THREADS="1 4 8 16"
export BH_MPI_RANKS="1 2 4 8"
bash scripts/benchmark.sh
```

En mode `--quick` les valeurs sont automatiquement réduites pour un
test rapide d'environ 15 minutes.