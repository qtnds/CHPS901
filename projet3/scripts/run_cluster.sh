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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

N=100000; ITERS=20
RES=results/cluster_${SLURM_JOB_ID}.csv
echo "timestamp,phase,version,n_particles,iters,n_ranks,n_threads,tree_us,force_us,integrate_us,total_us" > "$RES"

ts() { date '+%Y-%m-%dT%H:%M:%S'; }
run() {
    local phase=$1 ver=$2 n=$3 r=$4 t=$5
    shift 5
    local out; out=$("$@" 2>/dev/null || echo "TIMING tree_us=0 force_us=0 integrate_us=0 total_us=0")
    local line; line=$(echo "$out" | grep "^TIMING")
    local tree force integrate total
    tree=$(echo "$line"      | grep -oP 'tree_us=\K[0-9]+')
    force=$(echo "$line"     | grep -oP 'force_us=\K[0-9]+')
    integrate=$(echo "$line" | grep -oP 'integrate_us=\K[0-9]+')
    total=$(echo "$line"     | grep -oP 'total_us=\K[0-9]+')
    echo "$(ts),$phase,$ver,$n,$ITERS,$r,$t,$tree,$force,$integrate,$total" >> "$RES"
    echo "[$phase] $ver  N=$n R=$r T=$t → $total us"
}

echo "=== SLURM Job $SLURM_JOB_ID ==="
echo "Nœuds=$SLURM_NNODES  Rangs=$SLURM_NTASKS  Threads=$OMP_NUM_THREADS"

# Séquentiel
run seq seq $N 1 1 srun -n 1 --cpus-per-task=1 ./bin/bh_seq $N $ITERS 1

# OMP (toutes versions)
for VER in v1 v2 v3 v3s; do
    [[ -x ./bin/bh_omp_$VER ]] || continue
    run omp $VER $N 1 $OMP_NUM_THREADS \
        srun -n 1 --cpus-per-task=$OMP_NUM_THREADS \
        ./bin/bh_omp_$VER $N $ITERS $OMP_NUM_THREADS
done

# MPI (toutes versions, plusieurs configs rangs×threads)
for VER in v1 v2 v3; do
    [[ -x ./bin/bh_mpi_$VER ]] || continue
    for R in 1 2 4 $SLURM_NTASKS; do
        for T in 1 $OMP_NUM_THREADS; do
            run mpi $VER $N $R $T \
                srun --mpi=pmix -n $R --cpus-per-task=$T \
                ./bin/bh_mpi_$VER $N $ITERS $T
        done
    done
done

echo "=== Résultats → $RES ==="
