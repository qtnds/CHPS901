# ==============================================================
#  Makefile — Barnes-Hut N-body  (OpenMP + MPI)
#
#  Cibles :
#    make all          → compile les 4 variantes
#    make seq          → séquentiel (référence)
#    make omp          → OpenMP seulement
#    make mpi          → MPI seulement
#    make hybrid       → OpenMP + MPI
#    make clean        → supprime les binaires et résultats
#    make benchmark    → lance le script de benchmark complet
#    make benchmark_quick → benchmark rapide (debug / CI)
# ==============================================================

CXX      := g++
MPICXX   := mpicxx

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra
OMPFLAGS := -fopenmp
MPIFLAGS := -DUSE_MPI

SRC      := src/BH_parallel.cpp
BIN_DIR  := bin
RES_DIR  := results

.PHONY: all seq omp mpi hybrid clean benchmark benchmark_quick dirs

all: dirs seq omp mpi hybrid

dirs:
	@mkdir -p $(BIN_DIR) $(RES_DIR)

# ── Variante 1 : séquentielle (ni OMP ni MPI) ─────────────────
seq: dirs
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/bh_seq $(SRC)
	@echo "[OK] bin/bh_seq"

# ── Variante 2 : OpenMP uniquement ────────────────────────────
omp: dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $(BIN_DIR)/bh_omp $(SRC)
	@echo "[OK] bin/bh_omp"

# ── Variante 3 : MPI uniquement ───────────────────────────────
mpi: dirs
	$(MPICXX) $(CXXFLAGS) $(MPIFLAGS) -o $(BIN_DIR)/bh_mpi $(SRC)
	@echo "[OK] bin/bh_mpi"

# ── Variante 4 : hybride OpenMP + MPI ─────────────────────────
hybrid: dirs
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIFLAGS) -o $(BIN_DIR)/bh_hybrid $(SRC)
	@echo "[OK] bin/bh_hybrid"

clean:
	rm -rf $(BIN_DIR) $(RES_DIR)

benchmark: all
	bash scripts/benchmark.sh

benchmark_quick: all
	bash scripts/benchmark.sh --quick
