# Barnes-Hut N-body — Version parallèle (OpenMP + MPI)

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Installation des dépendances](#installation-des-dépendances)
4. [Compilation](#compilation)
5. [Exécution locale](#exécution-locale)
6. [Benchmark local](#benchmark-local)
7. [Portage sur cluster (SLURM)](#portage-sur-cluster-slurm)
8. [Stratégie de parallélisation](#stratégie-de-parallélisation)
9. [Résultats et analyse](#résultats-et-analyse)
10. [Pistes d'amélioration](#pistes-damélioration)

---

## Vue d'ensemble

Ce projet parallélise la simulation N-body de Barnes-Hut (`BH_parallel.cpp`) avec :

| Technologie | Rôle |
|------------|------|
| **OpenMP** | Parallélisme de boucles intra-nœud (mémoire partagée) |
| **MPI** | Décomposition en domaines inter-nœuds (mémoire distribuée) |
| **Hybride** | OpenMP à l'intérieur de chaque rang MPI |

Quatre binaires sont produits : `bh_seq`, `bh_omp`, `bh_mpi`, `bh_hybrid`.

---

## Structure du projet

```
bh_parallel/
├── src/
│   └── BH_parallel.cpp      # Code source unique (4 modes via #ifdef)
├── scripts/
│   ├── benchmark.sh         # Benchmark automatique local
│   ├── run_cluster.sh       # Script SLURM pour cluster
│   └── analyse_results.py   # Analyse et affichage des CSV
├── bin/                     # Binaires (créé par make)
├── results/                 # Fichiers CSV de résultats (créé par make)
└── Makefile
```

---

## Installation des dépendances

### En local (Ubuntu/Debian)

```bash
# Compilateur + OpenMP (inclus dans GCC)
sudo apt-get update
sudo apt-get install -y g++ make

# OpenMPI
sudo apt-get install -y libopenmpi-dev openmpi-bin

# Python (pour l'analyse des résultats)w
sudo apt-get install -y python3
```

### Vérifier les installations

```bash
g++ --version           # >= 9 recommandé
mpicxx --version        # doit afficher la version d'OpenMPI/MPICH
mpirun --version
echo '#include <omp.h>
int main(){return 0;}' | g++ -fopenmp -x c++ - -o /dev/null && echo "OMP OK"
```

### Sur cluster HPC (modules)

```bash
# Exemple sur cluster avec Lmod
module load gcc/12.2
module load openmpi/4.1.4

# Vérifier
mpicxx --showme       # doit lister les flags -I et -L
```

---

## Compilation

```bash
# Tout compiler (les 4 variantes)
make all

# Variante individuelle
make seq      # séquentiel — référence
make omp      # OpenMP uniquement
make mpi      # MPI uniquement
make hybrid   # OpenMP + MPI (recommandé sur cluster)

# Nettoyer
make clean
```

Les binaires se trouvent dans `bin/`.

---

## Exécution locale

Signature des programmes :

```
./bin/bh_<variante>  [n_particles]  [n_iters]  [n_threads_omp]
```

| Argument | Défaut | Description |
|---------|--------|-------------|
| `n_particles` | 10 000 | Nombre de particules |
| `n_iters` | 10 | Nombre d'itérations |
| `n_threads_omp` | 1 | Threads OpenMP par rang MPI |

### Exemples

```bash
# Séquentiel — 50 000 particules, 10 itérations
./bin/bh_seq 50000 10

# OpenMP — 4 threads
./bin/bh_omp 50000 10 4

# MPI — 2 rangs (chacun 1 thread)
mpirun -n 2 ./bin/bh_mpi 50000 10 1

# Hybride — 2 rangs × 2 threads = 4 « cœurs »
mpirun -n 2 -x OMP_NUM_THREADS=2 ./bin/bh_hybrid 50000 10 2
```

> **Sur 2 cœurs (votre machine locale)** : utilisez `mpirun -n 2 --oversubscribe`
> pour tester MPI même avec seulement 2 cœurs physiques. Le flag `--oversubscribe`
> autorise plus de rangs que de cœurs (utile pour le développement, pas pour
> la performance réelle).

---

## Benchmark local

```bash
# Benchmark complet (peut prendre 10-20 min selon la machine)
make benchmark

# Benchmark rapide (N réduit, pour vérifier que tout tourne)
make benchmark_quick
```

Les résultats sont écrits dans `results/` :

| Fichier | Contenu |
|--------|---------|
| `bench_seq.csv` | Temps séquentiel par taille de problème |
| `bench_omp.csv` | Temps + speedup OpenMP |
| `bench_mpi.csv` | Temps + speedup MPI |
| `bench_hybrid.csv` | Temps + speedup hybride |
| `summary.csv` | Consolidation de tout |

### Analyser les résultats

```bash
python3 scripts/analyse_results.py
# ou
python3 scripts/analyse_results.py --csv results/summary.csv
```

---

## Portage sur cluster (SLURM)

### 1. Copier le projet sur le cluster

```bash
# Depuis votre machine locale
scp -r bh_parallel/ <user>@<cluster>:~/

# ou via git
git clone <repo> && scp -r bh_parallel/ <user>@<cluster>:~/
```

### 2. Adapter le script SLURM

Éditer `scripts/run_cluster.sh` :

```bash
# Adapter les lignes suivantes selon votre cluster :
#SBATCH --nodes=4               # nombre de nœuds
#SBATCH --ntasks-per-node=1     # rangs MPI par nœud
#SBATCH --cpus-per-task=8       # threads OpenMP par rang
#SBATCH --partition=compute     # nom de la partition

# Charger les bons modules (décommenter et adapter) :
module load gcc/12 openmpi/4.1
```

### 3. Compiler sur le cluster (nœud de login)

```bash
cd ~/bh_parallel
make all
```

### 4. Soumettre le job

```bash
sbatch scripts/run_cluster.sh

# Suivre l'état
squeue -u $USER

# Voir la sortie en direct
tail -f results/slurm_<JOBID>.out
```

### 5. Récupérer les résultats

```bash
# Depuis votre machine locale
scp <user>@<cluster>:~/bh_parallel/results/*.csv results/
python3 scripts/analyse_results.py
```

### Recommandations configuration cluster

| Taille problème | Rangs MPI | Threads/rang | Commentaire |
|----------------|-----------|--------------|-------------|
| N ≤ 10 000 | 1 | = nb cœurs | OpenMP pur suffisant |
| 10 000 < N ≤ 100 000 | nb nœuds | 4–8 | Hybride recommandé |
| N > 100 000 | nb nœuds × 2 | 4–8 | Augmenter les rangs |

---

## Stratégie de parallélisation

### OpenMP — Justification des choix

#### Boucle de calcul de forces (`calc_interaction`)

```cpp
#pragma omp parallel for schedule(dynamic, 32)
for (std::size_t i = 0; i < n_local; ++i)
    calc_interaction(local_particles[i], &tree, theta);
```

- **Pourquoi `dynamic` ?** Le nombre d'opérations de `calc_interaction` par
  particule varie en fonction de la position relative dans l'arbre (certaines
  particules sont proches d'un nœud interne, d'autres le traversent en
  profondeur). Avec `static`, les threads finissent à des instants très
  différents → temps mort. `dynamic` rééquilibre dynamiquement.
- **chunk=32** : compromis entre overhead de scheduling (petit chunk = beaucoup
  de lock sur la file de travail) et granularité (grand chunk = déséquilibre
  resurgit).
- **Pas de race condition** : l'arbre est en lecture seule ; chaque particule
  i modifie uniquement `local_particles[i]` → indépendance totale.

#### Boucle d'intégration (`accelerate` + `move`)

```cpp
#pragma omp parallel for schedule(static)
for (std::size_t i = 0; i < n_local; ++i)
    local_particles[i].accelerate(dt).move(dt);
```

- **Pourquoi `static` ?** Coût parfaitement uniforme par particule
  (opérations arithmétiques simples). `static` est optimal : zéro overhead
  de scheduling.

#### Construction du qtree — NON parallélisée

La construction du qtree est **intentionnellement séquentielle**. Les
insertions modifient la structure arborescente de façon récursive ; les rendre
thread-safe nécessiterait des verrous qui annuleraient le gain. Son coût est
O(N log N), soit ≈ 5–15 % du temps total → ne vaut pas la complexité.

### MPI — Décomposition en domaines

L'espace [-0.5, 0.5]² est découpé en **bandes horizontales** (axe y), une
bande par rang. Chaque rang possède les particules dont la coordonnée y tombe
dans sa bande.

```
Rang 0 : y ∈ [-0.5, -0.25)
Rang 1 : y ∈ [-0.25, 0.0)
Rang 2 : y ∈ [0.0,   0.25)
Rang 3 : y ∈ [0.25,  0.5]
```

**Déroulement d'une itération MPI :**

```
1. MPI_Allgatherv   → chaque rang reçoit toutes les particules
                       (nécessaire pour construire l'arbre global)
2. Construction qtree global  (séquentielle, locale)
3. calc_interaction sur les particules LOCALES (→ OpenMP)
4. accelerate + move (→ OpenMP)
5. MPI_Alltoallv    → redistribution des particules migrantes
6. MPI_Barrier      → synchronisation
```

**Niveau MPI Thread :** `MPI_THREAD_FUNNELED` — seul le thread maître OpenMP
appelle MPI ; les threads OpenMP ne font que du calcul pur.

**Complexité communication :**
- `Allgatherv` : O(N × P) messages par itération. Pour de grands N, ceci
  devient le goulot d'étranglement → voir [Pistes d'amélioration](#pistes-damélioration).

---

## Résultats et analyse

Après `make benchmark` et `python3 scripts/analyse_results.py`, vous obtenez :

- **Speedup OpenMP** : attendu proche de 2x sur 2 cœurs physiques
  (limité par le faible nombre de cœurs en local)
- **Efficacité parallèle** : speedup / nb_processeurs × 100 %
  (idéal = 100 %, typiquement 60–80 % en pratique pour BH)
- **Meilleure configuration par N** : permet de choisir rapidement la
  config optimale selon la taille du problème

---

## Pistes d'amélioration

Les points suivants sont des améliorations à implémenter dans une prochaine étape :

1. **Réduction de la communication MPI** : au lieu d'un `Allgatherv` complet
   à chaque itération, envoyer uniquement les **centres de masse des sous-arbres**
   de chaque rang, puis construire un arbre approximatif global (far-field tree).
   Complexité communication réduite de O(N) à O(P × log N).

2. **Décomposition ORB (Orthogonal Recursive Bisection)** : remplacer la
   découpe en bandes par une décomposition récursive qui équilibre la charge
   (nombre de particules, non l'espace géométrique).

3. **Vectorisation SIMD** : annoter les boucles de force avec
   `#pragma omp simd` ou utiliser des intrinsics AVX2 pour le calcul de
   distance et de force.

4. **Tree walk parallèle** : construire le qtree en parallèle via un
   algorithme de construction concurrente (lock-free ou par régions).

5. **Équilibrage de charge dynamique** : adapter la taille des bandes MPI
   à chaque itération en fonction de la densité locale de particules.
