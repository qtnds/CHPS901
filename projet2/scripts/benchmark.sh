#!/usr/bin/env bash
# ==============================================================
#  benchmark.sh  — Étude itérative OMP puis MPI
#
#  Phase 1 : benchmark les 4 variantes OMP (v1, v2, v3, v3s)
#             → détermine la meilleure stratégie OMP
#  Phase 2 : benchmark les 3 variantes MPI (v1, v2, v3)
#             chacune compilée avec le meilleur OMP (dynamic=v2)
#
#  Usage :
#    bash scripts/benchmark.sh           # complet
#    bash scripts/benchmark.sh --quick   # N/iter réduits
#
#  Sorties dans results/ :
#    omp_results.csv
#    mpi_results.csv
#    seq_results.csv
# ==============================================================

set -euo pipefail

QUICK=0
for a in "$@"; do [[ "$a" == "--quick" ]] && QUICK=1; done

BIN=./bin
RES=./results
mkdir -p "$RES"

REPEATS=3   # médiane sur N runs

# ── Paramètres selon le mode ──────────────────────────────────
if [[ $QUICK -eq 1 ]]; then
    PARTICLES=(2000 5000 10000)
    ITERS=5
    OMP_THREADS=(1 2)
    MPI_RANKS=(1 2)
else
    PARTICLES=(5000 10000 50000 100000)
    ITERS=10
    OMP_THREADS=(1 2 4 8)
    MPI_RANKS=(1 2 4)
fi

# ── Helpers ───────────────────────────────────────────────────
ts()  { date '+%Y-%m-%dT%H:%M:%S'; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Extrait total_us depuis la ligne TIMING
parse_us() {
    grep -oP 'total_us=\K[0-9]+' | head -1 || echo "999999999"
}
parse_field() {
    local field="$1"
    grep -oP "${field}=\K[0-9]+" | head -1 || echo "0"
}

# Lance cmd $REPEATS fois, retourne médiane et tous les champs TIMING
run_median() {
    local -a cmd=("$@")
    local vals_total=() vals_tree=() vals_force=() vals_int=()
    for _ in $(seq 1 $REPEATS); do
        local out
        out=$("${cmd[@]}" 2>/dev/null || echo "TIMING tree_us=0 force_us=0 integrate_us=0 total_us=999999999")
        local line; line=$(echo "$out" | grep "^TIMING" || echo "TIMING tree_us=0 force_us=0 integrate_us=0 total_us=999999999")
        vals_total+=( "$(echo "$line" | parse_field total_us)" )
        vals_tree+=( "$(echo "$line"  | parse_field tree_us)" )
        vals_force+=( "$(echo "$line" | parse_field force_us)" )
        vals_int+=( "$(echo "$line"   | parse_field integrate_us)" )
    done
    median() { IFS=$'\n' local s=($(sort -n <<<"${*}")); unset IFS; echo "${s[$((${#s[@]}/2))]}"; }
    MED_TOTAL=$(median "${vals_total[@]}")
    MED_TREE=$(median  "${vals_tree[@]}")
    MED_FORCE=$(median "${vals_force[@]}")
    MED_INT=$(median   "${vals_int[@]}")
}

# ══════════════════════════════════════════════════════════════
#  SÉQUENTIEL — référence
# ══════════════════════════════════════════════════════════════
SEQ_CSV="$RES/seq_results.csv"
echo "timestamp,n_particles,iters,tree_us,force_us,integrate_us,total_us" > "$SEQ_CSV"

log "=== Benchmark séquentiel ==="
declare -A SEQ_REF   # SEQ_REF[N] = total_us de référence
for N in "${PARTICLES[@]}"; do
    run_median "$BIN/bh_seq" "$N" "$ITERS" 1
    SEQ_REF[$N]=$MED_TOTAL
    echo "$(ts),$N,$ITERS,$MED_TREE,$MED_FORCE,$MED_INT,$MED_TOTAL" >> "$SEQ_CSV"
    log "  seq N=$N → ${MED_TOTAL} us"
done

# ══════════════════════════════════════════════════════════════
#  PHASE 1 : OpenMP
# ══════════════════════════════════════════════════════════════
OMP_CSV="$RES/omp_results.csv"
echo "timestamp,version,n_particles,iters,n_threads,tree_us,force_us,integrate_us,total_us,speedup_vs_seq" \
     > "$OMP_CSV"

log ""
log "=== Phase 1 : Benchmark OpenMP ==="

for VER in v1 v2 v3 v3s; do
    BIN_PATH="$BIN/bh_omp_${VER}"
    [[ -x "$BIN_PATH" ]] || { log "  [SKIP] $BIN_PATH manquant"; continue; }
    for N in "${PARTICLES[@]}"; do
        REF=${SEQ_REF[$N]}
        for T in "${OMP_THREADS[@]}"; do
            run_median "$BIN_PATH" "$N" "$ITERS" "$T"
            SPD=$(awk "BEGIN{printf \"%.4f\", $REF/$MED_TOTAL}")
            echo "$(ts),$VER,$N,$ITERS,$T,$MED_TREE,$MED_FORCE,$MED_INT,$MED_TOTAL,$SPD" >> "$OMP_CSV"
            log "  omp_${VER}  N=$N T=$T → ${MED_TOTAL} us  (speedup ${SPD}x)"
        done
    done
done

# ── Trouver la meilleure version OMP (T=max, N=max) ───────────
MAX_T="${OMP_THREADS[-1]}"
MAX_N="${PARTICLES[-1]}"
BEST_OMP_VER=""
BEST_OMP_US=999999999
while IFS=',' read -r _ts ver n _it t _tr _f _i total _spd; do
    [[ "$ver" == "version" ]] && continue
    [[ "$n"   == "$MAX_N"  ]] || continue
    [[ "$t"   == "$MAX_T"  ]] || continue
    if (( total < BEST_OMP_US )); then
        BEST_OMP_US=$total
        BEST_OMP_VER=$ver
    fi
done < "$OMP_CSV"

log ""
log ">>> Meilleure version OMP : omp_${BEST_OMP_VER} (${BEST_OMP_US} us @ N=${MAX_N}, T=${MAX_T})"
log ""

# ══════════════════════════════════════════════════════════════
#  PHASE 2 : MPI (avec le meilleur OMP = dynamic = v2 dans le binaire)
# ══════════════════════════════════════════════════════════════
MPI_CSV="$RES/mpi_results.csv"
echo "timestamp,version,n_particles,iters,n_ranks,n_threads,tree_us,force_us,integrate_us,total_us,speedup_vs_seq" \
     > "$MPI_CSV"

log "=== Phase 2 : Benchmark MPI ==="
log "    (tous les binaires MPI utilisent OMP dynamic=v2 en interne)"

for VER in v1 v2 v3; do
    BIN_PATH="$BIN/bh_mpi_${VER}"
    [[ -x "$BIN_PATH" ]] || { log "  [SKIP] $BIN_PATH manquant"; continue; }
    for N in "${PARTICLES[@]}"; do
        REF=${SEQ_REF[$N]}
        for R in "${MPI_RANKS[@]}"; do
            for T in "${OMP_THREADS[@]}"; do
                run_median \
                    mpirun --oversubscribe -n "$R" \
                    -x OMP_NUM_THREADS="$T" \
                    "$BIN_PATH" "$N" "$ITERS" "$T"
                SPD=$(awk "BEGIN{printf \"%.4f\", $REF/$MED_TOTAL}")
                echo "$(ts),$VER,$N,$ITERS,$R,$T,$MED_TREE,$MED_FORCE,$MED_INT,$MED_TOTAL,$SPD" >> "$MPI_CSV"
                log "  mpi_${VER}  N=$N R=$R T=$T → ${MED_TOTAL} us  (speedup ${SPD}x)"
            done
        done
    done
done

# ── Résumé final ──────────────────────────────────────────────
log ""
log "=========================================="
log "  Benchmark terminé."
log "  Résultats :"
ls -lh "$RES"/*.csv
log "  Lancer l'analyse : python3 scripts/analyse.py"
log "=========================================="
