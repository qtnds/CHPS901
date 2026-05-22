#!/bin/bash
# ==============================================================
#  run_cluster.sh — Soumettre sur cluster SLURM
#  Adapter les lignes #SBATCH selon votre environnement.
# ==============================================================
#SBATCH --job-name=bh_parallel
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --account=r250123
#SBATCH --constraint=armgpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cores=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00


romeo_load_armgpu_env
spack load gcc@11.4.1 /nrzc6ai
spack load openmpi@4.1.7 /nkokjyt
spack load cuda@12.6.2 /3mzltpz

# ── Sécurité ──────────────────────────────────────────────────
set -euo pipefail

# ── Répertoire de travail ─────────────────────────────────────
# SLURM_SUBMIT_DIR = répertoire depuis lequel sbatch est lancé
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results bin

# ── Modules ───────────────────────────────────────────────────
# Décommenter selon l'environnement Romeo :
# module purge
# module load gcc/12.2
# module load openmpi/4.1.4
# module load cuda/12.0      # décommenter pour --phase 4 avec GPU réel

# ── Bannière ──────────────────────────────────────────────────
echo "============================================"
echo "  BH Benchmark — SLURM job ${SLURM_JOB_ID:-local}"
echo "  Nœud       : $(hostname)"
echo "  Début      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Partition  : ${SLURM_JOB_PARTITION:-N/A}"
echo "  ntasks     : ${SLURM_NTASKS:-1}"
echo "  cpus/task  : ${SLURM_CPUS_PER_TASK:-1}"
echo "  Submit dir : $(pwd)"
echo "============================================"
echo ""

# ── Compilation ───────────────────────────────────────────────
echo "--- Compilation ---"
make all
echo ""

# ── Paramètres MPI/OMP calés sur l'allocation SLURM ──────────
# On construit les listes en puissances de 2 jusqu'au max alloué.
N_RANKS_MAX=${SLURM_NTASKS:-4}
N_THREADS_MAX=${SLURM_CPUS_PER_TASK:-8}

build_pow2_list() {
    local max=$1
    local list="1"; local v=2
    while [[ $v -le $max ]]; do
        list="$list $v"; v=$((v * 2))
    done
    # Ajouter le max exact s'il n'est pas déjà une puissance de 2
    [[ "$list" == *" $max"* || "$list" == "1" && $max -eq 1 ]] \
        || list="$list $max"
    echo "$list"
}

OMP_LIST=$(build_pow2_list "$N_THREADS_MAX")
MPI_LIST=$(build_pow2_list "$N_RANKS_MAX")

echo "MPI_RANKS    : $MPI_LIST"
echo "OMP_THREADS  : $OMP_LIST"
echo ""

# ── Variables d'environnement pour benchmark.sh ───────────────
export BH_OMP_THREADS="$OMP_LIST"
export BH_MPI_RANKS="$MPI_LIST"
export BH_PARTICLES="5000 10000 50000 100000 500000"
export BH_PARTICLES_CUDA="1000 5000 10000 50000 100000 500000"
export BH_ITERS=10

# Affinité OMP : chaque rang MPI dispose de cpus-per-task cœurs
export OMP_NUM_THREADS=$N_THREADS_MAX
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ── Benchmark ─────────────────────────────────────────────────
echo "--- Lancement benchmark.sh $* ---"
echo ""
bash scripts/benchmark.sh "$@"

# ── Analyse ───────────────────────────────────────────────────
echo ""
echo "--- Génération des figures ---"
python3 scripts/analyse.py --results results/ \
    && echo "  → results/figures/" \
    || echo "[WARN] analyse.py échoué (matplotlib absent ?)"

# ── Résumé ────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Fin : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  CSVs produits :"
ls -lh results/*.csv 2>/dev/null | awk '{print "    "$NF" ("$5")"}' || true
echo "  Figures :"
ls results/figures/*.png 2>/dev/null | wc -l \
    | xargs -I{} echo "    {} figures dans results/figures/"
echo "============================================"