#!/usr/bin/env bash
# ==============================================================
#  benchmark.sh — Benchmark des 4 variantes de BH N-body
#
#  Usage :
#    bash scripts/benchmark.sh          # complet
#    bash scripts/benchmark.sh --quick  # rapide (N et iter réduits)
#
#  Sorties :
#    results/bench_seq.csv
#    results/bench_omp.csv
#    results/bench_mpi.csv
#    results/bench_hybrid.csv
#    results/summary.csv
# ==============================================================

set -euo pipefail

QUICK=0
for arg in "$@"; do [[ "$arg" == "--quick" ]] && QUICK=1; done

BIN=./bin
RES=./results
mkdir -p "$RES"

# ── Paramètres ──────────────────────────────────────────────
if [[ $QUICK -eq 1 ]]; then
    PARTICLES=(1000 5000)
    ITERS=5
    OMP_THREADS=(1 2)
    MPI_RANKS=(1 2)
else
    PARTICLES=(10000 50000 100000)
    ITERS=10
    OMP_THREADS=(1 2 4 8)
    MPI_RANKS=(1 2 4)
fi

THETA=0.5
REPEATS=3        # nombre de répétitions pour moyenner

# ── Helpers ────────────────────────────────────────────────

timestamp() { date '+%Y-%m-%dT%H:%M:%S'; }

# Extrait le TOTAL en microsecondes depuis stdout du programme.
# Cherche la ligne "TOTAL : XXXX us"
parse_total_us() {
    grep -oP 'TOTAL\s*:\s*\K[0-9]+' || echo "0"
}

# Lance une commande $REPEATS fois et renvoie la médiane en µs.
run_and_median() {
    local cmd=("$@")
    local vals=()
    for _ in $(seq 1 $REPEATS); do
        local out
        out=$("${cmd[@]}" 2>/dev/null)
        local v
        v=$(echo "$out" | parse_total_us)
        vals+=("$v")
    done
    # Médiane (tri + sélection)
    IFS=$'\n' sorted=($(sort -n <<<"${vals[*]}")); unset IFS
    echo "${sorted[$((REPEATS/2))]}"
}

# ── Benchmark séquentiel ────────────────────────────────────
echo "=== Benchmark séquentiel ==="
CSV_SEQ="$RES/bench_seq.csv"
echo "timestamp,n_particles,iters,total_us" > "$CSV_SEQ"

for N in "${PARTICLES[@]}"; do
    echo -n "  N=$N ... "
    t=$(run_and_median "$BIN/bh_seq" "$N" "$ITERS" 1)
    echo "$t us"
    echo "$(timestamp),$N,$ITERS,$t" >> "$CSV_SEQ"
done

# ── Benchmark OpenMP ────────────────────────────────────────
echo "=== Benchmark OpenMP ==="
CSV_OMP="$RES/bench_omp.csv"
echo "timestamp,n_particles,iters,n_threads,total_us,speedup_vs_seq" > "$CSV_OMP"

for N in "${PARTICLES[@]}"; do
    # Récupérer la référence séquentielle (1 thread)
    ref=$(run_and_median "$BIN/bh_omp" "$N" "$ITERS" 1)
    for T in "${OMP_THREADS[@]}"; do
        echo -n "  N=$N T=$T ... "
        t=$(run_and_median "$BIN/bh_omp" "$N" "$ITERS" "$T")
        speedup=$(awk "BEGIN { printf \"%.3f\", $ref/$t }")
        echo "$t us  (speedup vs T=1: ${speedup}x)"
        echo "$(timestamp),$N,$ITERS,$T,$t,$speedup" >> "$CSV_OMP"
    done
done

# ── Benchmark MPI ───────────────────────────────────────────
echo "=== Benchmark MPI ==="
CSV_MPI="$RES/bench_mpi.csv"
echo "timestamp,n_particles,iters,n_ranks,total_us,speedup_vs_seq" > "$CSV_MPI"

# Référence séquentielle (déjà calculée mais on la relit)
for N in "${PARTICLES[@]}"; do
    ref_line=$(grep ",$N," "$CSV_SEQ" | tail -1)
    ref_us=$(echo "$ref_line" | cut -d',' -f4)
    for R in "${MPI_RANKS[@]}"; do
        echo -n "  N=$N ranks=$R ... "
        t=$(run_and_median mpirun --oversubscribe -n "$R" "$BIN/bh_mpi" "$N" "$ITERS" 1)
        speedup=$(awk "BEGIN { printf \"%.3f\", $ref_us/$t }")
        echo "$t us  (speedup vs seq: ${speedup}x)"
        echo "$(timestamp),$N,$ITERS,$R,$t,$speedup" >> "$CSV_MPI"
    done
done

# ── Benchmark hybride OpenMP+MPI ────────────────────────────
echo "=== Benchmark Hybride ==="
CSV_HYB="$RES/bench_hybrid.csv"
echo "timestamp,n_particles,iters,n_ranks,n_threads,total_us,speedup_vs_seq" > "$CSV_HYB"

for N in "${PARTICLES[@]}"; do
    ref_line=$(grep ",$N," "$CSV_SEQ" | tail -1)
    ref_us=$(echo "$ref_line" | cut -d',' -f4)
    for R in "${MPI_RANKS[@]}"; do
        for T in "${OMP_THREADS[@]}"; do
            echo -n "  N=$N ranks=$R threads=$T ... "
            t=$(run_and_median \
                mpirun --oversubscribe -n "$R" \
                    -x OMP_NUM_THREADS="$T" \
                    "$BIN/bh_hybrid" "$N" "$ITERS" "$T")
            speedup=$(awk "BEGIN { printf \"%.3f\", $ref_us/$t }")
            echo "$t us  (speedup vs seq: ${speedup}x)"
            echo "$(timestamp),$N,$ITERS,$R,$T,$t,$speedup" >> "$CSV_HYB"
        done
    done
done

# ── Résumé consolidé ────────────────────────────────────────
echo "=== Génération du résumé ==="
CSV_SUM="$RES/summary.csv"
{
echo "variant,n_particles,iters,n_ranks,n_threads,total_us,speedup"
while IFS=',' read -r ts n it total; do
    [[ "$ts" == "timestamp" ]] && continue
    echo "seq,$n,$it,1,1,$total,1.000"
done < "$CSV_SEQ"

while IFS=',' read -r ts n it t total spd; do
    [[ "$ts" == "timestamp" ]] && continue
    echo "omp,$n,$it,1,$t,$total,$spd"
done < "$CSV_OMP"

while IFS=',' read -r ts n it r total spd; do
    [[ "$ts" == "timestamp" ]] && continue
    echo "mpi,$n,$it,$r,1,$total,$spd"
done < "$CSV_MPI"

while IFS=',' read -r ts n it r t total spd; do
    [[ "$ts" == "timestamp" ]] && continue
    echo "hybrid,$n,$it,$r,$t,$total,$spd"
done < "$CSV_HYB"
} > "$CSV_SUM"

echo ""
echo "============================================"
echo "  Benchmark terminé. Résultats dans $RES/"
echo "  Fichiers produits :"
ls -lh "$RES"/*.csv
echo "============================================"
