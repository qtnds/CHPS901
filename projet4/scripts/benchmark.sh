#!/usr/bin/env bash
# ==============================================================
#  benchmark.sh — Étude itérative complète BH parallèle
#
#  Phase 1 : OMP (v1 v2 v3 v3s)   — baseline : bh_seq
#  Phase 2 : MPI × OMP (v1 v2 v3) — baseline : bh_seq
#  Phase 3 : Hybride final         — baseline : bh_seq
#  Phase 4 : CUDA — balayage étendu de N pour voir le crossover
#
#  Usage :
#    bash scripts/benchmark.sh             # complet
#    bash scripts/benchmark.sh --quick     # N/iter réduits
#    bash scripts/benchmark.sh --phase 2   # une seule phase
#
#  Variables d'environnement :
#    BH_PARTICLES       ex: "10000 50000 100000"
#    BH_PARTICLES_CUDA  ex: "1000 5000 10000 50000 100000 500000"
#    BH_ITERS           ex: 20
#    BH_OMP_THREADS     ex: "1 4 8"
#    BH_MPI_RANKS       ex: "1 2 4 8"
# ==============================================================

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────
QUICK=0; PHASE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)  QUICK=1;    shift ;;
        --phase)  PHASE=$2;   shift 2 ;;
        *)        shift ;;
    esac
done

BIN=./bin; RES=./results
mkdir -p "$RES"
REPEATS=3

# ── Paramètres par défaut ─────────────────────────────────────
if [[ $QUICK -eq 1 ]]; then
    PARTICLES=(2000 10000)
    PARTICLES_CUDA=(1000 2000 5000 10000 50000)
    ITERS=2
    OMP_THREADS=(1 2)
    MPI_RANKS=(1 2)
else
    PARTICLES=(5000 10000 50000 100000)
    PARTICLES_CUDA=(1000 5000 10000 50000 100000 500000)
    ITERS=10
    OMP_THREADS=(1 2 4 8)
    MPI_RANKS=(1 2 4)
fi

# Surcharge via variables d'environnement
[[ -n "${BH_PARTICLES:-}"      ]] && read -ra PARTICLES      <<< "$BH_PARTICLES"
[[ -n "${BH_PARTICLES_CUDA:-}" ]] && read -ra PARTICLES_CUDA <<< "$BH_PARTICLES_CUDA"
[[ -n "${BH_ITERS:-}"          ]] && ITERS=$BH_ITERS
[[ -n "${BH_OMP_THREADS:-}"    ]] && read -ra OMP_THREADS    <<< "$BH_OMP_THREADS"
[[ -n "${BH_MPI_RANKS:-}"      ]] && read -ra MPI_RANKS      <<< "$BH_MPI_RANKS"

ts()  { date '+%Y-%m-%dT%H:%M:%S'; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] [ERR] $*" >&2; }

# ══════════════════════════════════════════════════════════════
#  run_mpirun N_RANKS N_THREADS BINARY [ARGS...]
# ══════════════════════════════════════════════════════════════
run_mpirun() {
    local n_ranks=$1 n_threads=$2; shift 2
    export OMP_NUM_THREADS=$n_threads
    mpirun -np "$n_ranks" --oversubscribe "$@"
}

# ══════════════════════════════════════════════════════════════
#  run_median CMD [ARGS...]
#
#  Exécute REPEATS fois, retourne la médiane des champs TIMING.
#  Champs : total_us tree_us force_us integrate_us
#           comm_us serial_us gpu_us
# ══════════════════════════════════════════════════════════════
run_median() {
    local -a tot=() tr=() fo=() it=() co=() se=() gu=()

    for _ in $(seq 1 $REPEATS); do
        local out rc=0
        local errfile; errfile=$(mktemp)
        out=$("$@" 2>"$errfile") || rc=$?

        if [[ $rc -ne 0 ]]; then
            err "Échec (rc=$rc) : $*"
            err "Stderr : $(head -3 "$errfile")"
            rm -f "$errfile"
            tot+=(999999999); tr+=(0); fo+=(0); it+=(0); co+=(0); se+=(0); gu+=(0)
            continue
        fi
        rm -f "$errfile"

        local line; line=$(echo "$out" | grep "^TIMING" || true)
        if [[ -z "$line" ]]; then
            err "Pas de ligne TIMING : $*"
            err "Sortie : $(echo "$out" | head -3)"
            tot+=(999999999); tr+=(0); fo+=(0); it+=(0); co+=(0); se+=(0); gu+=(0)
            continue
        fi

        pf() { echo "$line" | grep -oP "${1}=\K[0-9]+" || echo 0; }
        tot+=("$(pf total_us)")
        tr+=("$(pf tree_us)")
        fo+=("$(pf force_us)")
        it+=("$(pf integrate_us)")
        co+=("$(pf comm_us)")
        se+=("$(pf serial_us)")
        gu+=("$(pf gpu_us)")
    done

    med() { IFS=$'\n' local s=($(sort -n <<<"${*}")); unset IFS; echo "${s[$((${#s[@]}/2))]}"; }
    MED_TOTAL=$(med  "${tot[@]}")
    MED_TREE=$(med   "${tr[@]}")
    MED_FORCE=$(med  "${fo[@]}")
    MED_INT=$(med    "${it[@]}")
    MED_COMM=$(med   "${co[@]}")
    MED_SERIAL=$(med "${se[@]}")
    MED_GPU=$(med    "${gu[@]}")
}

# ══════════════════════════════════════════════════════════════
#  SÉQUENTIEL — bh_seq (BH_omp_v1 sans OMP, sortie TIMING)
#  Sert de baseline pour tous les speedups.
# ══════════════════════════════════════════════════════════════
run_seq() {
    [[ $PHASE -ne 0 && $PHASE -ne 1 ]] && return
    local CSV="$RES/seq_results.csv"
    echo "timestamp,n_particles,iters,tree_us,force_us,integrate_us,total_us" > "$CSV"
    log "=== Séquentiel ==="
    declare -gA SEQ_REF=()
    for N in "${PARTICLES[@]}"; do
        run_median "$BIN/bh_seq" "$N" "$ITERS" 1
        SEQ_REF[$N]=$MED_TOTAL
        echo "$(ts),$N,$ITERS,$MED_TREE,$MED_FORCE,$MED_INT,$MED_TOTAL" >> "$CSV"
        log "  seq  N=$N → ${MED_TOTAL} us  [tree=${MED_TREE} force=${MED_FORCE} integ=${MED_INT}]"
    done

    # Mesurer aussi sur PARTICLES_CUDA pour avoir la baseline au crossover
    local CSV2="$RES/seq_cuda_range.csv"
    echo "timestamp,n_particles,iters,tree_us,force_us,integrate_us,total_us" > "$CSV2"
    for N in "${PARTICLES_CUDA[@]}"; do
        local already=0
        for P in "${PARTICLES[@]}"; do [[ "$P" == "$N" ]] && already=1; done
        if [[ $already -eq 0 ]]; then
            run_median "$BIN/bh_seq" "$N" "$ITERS" 1
            SEQ_REF[$N]=$MED_TOTAL
            echo "$(ts),$N,$ITERS,$MED_TREE,$MED_FORCE,$MED_INT,$MED_TOTAL" >> "$CSV2"
            log "  seq  N=$N (CUDA range) → ${MED_TOTAL} us"
        fi
    done
}

# ══════════════════════════════════════════════════════════════
#  PHASE 1 : OpenMP
# ══════════════════════════════════════════════════════════════
run_omp() {
    [[ $PHASE -ne 0 && $PHASE -ne 1 ]] && return
    local CSV="$RES/omp_results.csv"
    echo "timestamp,version,n_particles,iters,n_threads,tree_us,force_us,integrate_us,total_us,speedup" > "$CSV"
    log ""; log "=== Phase 1 : OpenMP ==="
    for VER in v1 v2 v3 v3s; do
        local B="$BIN/bh_omp_${VER}"
        [[ -x "$B" ]] || { log "  [skip] $B absent"; continue; }
        for N in "${PARTICLES[@]}"; do
            local REF=${SEQ_REF[$N]:-999999999}
            for T in "${OMP_THREADS[@]}"; do
                run_median "$B" "$N" "$ITERS" "$T"
                local SPD; SPD=$(awk "BEGIN{printf \"%.4f\",$REF/$MED_TOTAL}")
                echo "$(ts),$VER,$N,$ITERS,$T,$MED_TREE,$MED_FORCE,$MED_INT,$MED_TOTAL,$SPD" >> "$CSV"
                log "  omp_${VER}  N=$N T=$T → ${MED_TOTAL} us (${SPD}x)"
            done
        done
    done
}

# ══════════════════════════════════════════════════════════════
#  PHASE 2 : MPI × OMP
# ══════════════════════════════════════════════════════════════
run_mpi() {
    [[ $PHASE -ne 0 && $PHASE -ne 2 ]] && return
    local CSV="$RES/mpi_results.csv"
    echo "timestamp,version,n_particles,iters,n_ranks,n_threads,tree_us,force_us,integrate_us,total_us,speedup" > "$CSV"
    log ""; log "=== Phase 2 : MPI × OMP ==="
    for VER in v1 v2 v3; do
        local B="$BIN/bh_mpi_${VER}"
        [[ -x "$B" ]] || { log "  [skip] $B absent"; continue; }
        for N in "${PARTICLES[@]}"; do
            local REF=${SEQ_REF[$N]:-999999999}
            for R in "${MPI_RANKS[@]}"; do
                for T in "${OMP_THREADS[@]}"; do
                    run_median run_mpirun "$R" "$T" "$B" "$N" "$ITERS" "$T"
                    local SPD; SPD=$(awk "BEGIN{printf \"%.4f\",$REF/$MED_TOTAL}")
                    echo "$(ts),$VER,$N,$ITERS,$R,$T,$MED_TREE,$MED_FORCE,$MED_INT,$MED_TOTAL,$SPD" >> "$CSV"
                    log "  mpi_${VER}  N=$N R=$R T=$T → ${MED_TOTAL} us (${SPD}x)"
                done
            done
        done
    done
}

# ══════════════════════════════════════════════════════════════
#  PHASE 3 : Hybride final
# ══════════════════════════════════════════════════════════════
run_hybrid() {
    [[ $PHASE -ne 0 && $PHASE -ne 3 ]] && return
    local B="$BIN/bh_hybrid"
    [[ -x "$B" ]] || { log "[skip] $B absent"; return; }
    local CSV="$RES/hybrid_results.csv"
    echo "timestamp,n_particles,iters,n_ranks,n_threads,tree_us,force_us,integrate_us,comm_us,total_us,speedup" > "$CSV"
    log ""; log "=== Phase 3 : Hybride final MPI+OMP ==="
    for N in "${PARTICLES[@]}"; do
        local REF=${SEQ_REF[$N]:-999999999}
        for R in "${MPI_RANKS[@]}"; do
            for T in "${OMP_THREADS[@]}"; do
                run_median run_mpirun "$R" "$T" "$B" "$N" "$ITERS" "$T"
                local SPD; SPD=$(awk "BEGIN{printf \"%.4f\",$REF/$MED_TOTAL}")
                echo "$(ts),$N,$ITERS,$R,$T,$MED_TREE,$MED_FORCE,$MED_INT,$MED_COMM,$MED_TOTAL,$SPD" >> "$CSV"
                log "  hybrid  N=$N R=$R T=$T → ${MED_TOTAL} us (${SPD}x)  [comm=${MED_COMM}us]"
            done
        done
    done
}

# ══════════════════════════════════════════════════════════════
#  PHASE 4 : CUDA — balayage étendu de N
#
#  PARTICLES_CUDA couvre une plage large (1k → 500k) pour observer
#  le crossover GPU/CPU. La colonne seq_us est insérée en relisant
#  SEQ_REF (ou en lançant bh_seq sur les N manquants).
# ══════════════════════════════════════════════════════════════
run_cuda() {
    [[ $PHASE -ne 0 && $PHASE -ne 4 ]] && return
    local B="$BIN/bh_cuda"
    [[ -x "$B" ]] || { log "[skip] $B absent"; return; }

    local CSV="$RES/cuda_results.csv"
    echo "timestamp,n_particles,iters,n_threads_cpu,tree_us,serial_us,gpu_us,integrate_us,total_us,seq_us,speedup" > "$CSV"
    log ""; log "=== Phase 4 : CUDA (balayage N étendu) ==="
    log "    N testé : ${PARTICLES_CUDA[*]}"

    for N in "${PARTICLES_CUDA[@]}"; do
        # Baseline séq : utiliser SEQ_REF si disponible, sinon mesurer bh_seq
        local REF=${SEQ_REF[$N]:-0}
        if [[ $REF -eq 0 ]]; then
            [[ -x "$BIN/bh_seq" ]] || { log "  [skip] bh_seq absent"; continue; }
            run_median "$BIN/bh_seq" "$N" "$ITERS" 1
            REF=$MED_TOTAL
            SEQ_REF[$N]=$REF
            log "  seq  N=$N → ${REF} us (mesuré pour crossover)"
        fi

        for T in "${OMP_THREADS[@]}"; do
            run_median "$B" "$N" "$ITERS" "$T"
            local SPD="0"
            [[ $REF -gt 0 ]] && SPD=$(awk "BEGIN{printf \"%.4f\",$REF/$MED_TOTAL}")
            echo "$(ts),$N,$ITERS,$T,$MED_TREE,$MED_SERIAL,$MED_GPU,$MED_INT,$MED_TOTAL,$REF,$SPD" >> "$CSV"
            log "  cuda  N=$N T=$T → ${MED_TOTAL} us (${SPD}x)  [tree=${MED_TREE} serial=${MED_SERIAL} gpu=${MED_GPU} integ=${MED_INT}]"
        done
    done
}

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
log "=== Benchmark BH parallèle ==="
log "PARTICLES=${PARTICLES[*]}"
log "PARTICLES_CUDA=${PARTICLES_CUDA[*]}"
log "ITERS=$ITERS  OMP_THREADS=${OMP_THREADS[*]}  MPI_RANKS=${MPI_RANKS[*]}"
log "REPEATS=$REPEATS  QUICK=$QUICK  PHASE=$PHASE"
log ""

run_seq
run_omp
run_mpi
run_hybrid
run_cuda

log ""
log "=============================="
log " Benchmark terminé."
log " CSVs dans : $RES/"
ls -lh "$RES"/*.csv 2>/dev/null | awk '{print "  "$NF" ("$5")"}' || true
log " Analyse : python3 scripts/analyse.py"
log "=============================="