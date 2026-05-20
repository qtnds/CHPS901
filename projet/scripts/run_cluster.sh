#!/bin/bash
# ==============================================================
#  run_cluster.sh — Script SLURM pour cluster multi-nœuds
#
#  Adapter les paramètres #SBATCH selon votre cluster.
#  Soumettre avec : sbatch scripts/run_cluster.sh
#
#  Produit : results/cluster_<SLURM_JOB_ID>.csv
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


# ── Charger les modules (adapter selon l'environnement HPC) ──
# module load gcc/12 openmpi/4.1
# module load intel-mpi/2021  # alternative Intel MPI

# ── Paramètres de simulation ─────────────────────────────────
N_PARTICLES=100000
ITERS=20
OMP_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$OMP_THREADS

OUTFILE="results/cluster_${SLURM_JOB_ID}.csv"
echo "variant,n_particles,iters,n_ranks,n_threads,node,total_us" > "$OUTFILE"

echo "=== Job SLURM $SLURM_JOB_ID ==="
echo "Nœuds : $SLURM_NNODES  Rangs : $SLURM_NTASKS  Threads/rang : $OMP_THREADS"
echo "N_PARTICLES=$N_PARTICLES  ITERS=$ITERS"

# ── Variante séquentielle (référence, 1 rang, 1 thread) ───────
echo -n "Séquentiel ... "
out=$(srun -n 1 --cpus-per-task=1 ./bin/bh_seq "$N_PARTICLES" "$ITERS" 1 2>/dev/null)
us=$(echo "$out" | grep -oP 'TOTAL\s*:\s*\K[0-9]+' || echo "0")
echo "$us us"
echo "seq,$N_PARTICLES,$ITERS,1,1,$(hostname),$us" >> "$OUTFILE"

# ── Variante OpenMP (1 rang, N threads) ───────────────────────
echo -n "OpenMP ($OMP_THREADS threads) ... "
out=$(srun -n 1 --cpus-per-task="$OMP_THREADS" \
    ./bin/bh_omp "$N_PARTICLES" "$ITERS" "$OMP_THREADS" 2>/dev/null)
us=$(echo "$out" | grep -oP 'TOTAL\s*:\s*\K[0-9]+' || echo "0")
echo "$us us"
echo "omp,$N_PARTICLES,$ITERS,1,$OMP_THREADS,$(hostname),$us" >> "$OUTFILE"

# ── Variante MPI (N rangs, 1 thread) ──────────────────────────
echo -n "MPI ($SLURM_NTASKS rangs) ... "
out=$(srun --mpi=pmix -n "$SLURM_NTASKS" --cpus-per-task=1 \
    ./bin/bh_mpi "$N_PARTICLES" "$ITERS" 1 2>/dev/null)
us=$(echo "$out" | grep -oP 'TOTAL\s*:\s*\K[0-9]+' || echo "0")
echo "$us us"
echo "mpi,$N_PARTICLES,$ITERS,$SLURM_NTASKS,1,$(hostname),$us" >> "$OUTFILE"

# ── Variante Hybride (N rangs × N threads) ────────────────────
echo -n "Hybride ($SLURM_NTASKS rangs × $OMP_THREADS threads) ... "
out=$(srun --mpi=pmix -n "$SLURM_NTASKS" --cpus-per-task="$OMP_THREADS" \
    ./bin/bh_hybrid "$N_PARTICLES" "$ITERS" "$OMP_THREADS" 2>/dev/null)
us=$(echo "$out" | grep -oP 'TOTAL\s*:\s*\K[0-9]+' || echo "0")
echo "$us us"
echo "hybrid,$N_PARTICLES,$ITERS,$SLURM_NTASKS,$OMP_THREADS,$(hostname),$us" >> "$OUTFILE"

echo ""
echo "=== Résultats dans $OUTFILE ==="
cat "$OUTFILE"
