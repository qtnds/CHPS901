# ==============================================================
#  Makefile — Barnes-Hut N-body  Étude de parallélisation
#
#  Arbre des variantes
#  ───────────────────
#  Référence
#    bh_seq          séquentiel pur
#
#  Phase 1 — OpenMP (3 stratégies d'ordonnancement)
#    bh_omp_v1       schedule(static)           sur la boucle forces
#    bh_omp_v2       schedule(dynamic, 32)       sur la boucle forces
#    bh_omp_v3       schedule(guided, 16)        sur la boucle forces
#    bh_omp_v3s      schedule(guided) + tri spatial (SPATIAL_SORT)
#
#  Phase 2 — MPI + meilleur OMP (dynamic,32 = v2)
#    bh_mpi_v1       bandes Y  + Allgather complet (7 floats/particule)
#    bh_mpi_v2       blocs idx + Allgather complet (pas de redistribution)
#    bh_mpi_v3       bandes Y  + Allgather léger   (3 floats/particule)
# ==============================================================

CXX    := g++
MPICXX := mpicxx

CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -I src
OMPFLAGS  := -fopenmp
MPIDEF    := -DUSE_MPI

BIN := bin
SRC := src

.PHONY: all seq omp mpi clean benchmark benchmark_quick dirs

all: dirs seq omp mpi

dirs:
	@mkdir -p $(BIN) results

# ── Séquentiel ────────────────────────────────────────────────
seq: dirs
	$(CXX) $(CXXFLAGS) -o $(BIN)/bh_seq $(SRC)/BH_omp_v1.cpp
	@echo "[OK] bh_seq"

# ── Phase 1 : OpenMP ──────────────────────────────────────────
omp: dirs \
  $(BIN)/bh_omp_v1 $(BIN)/bh_omp_v2 \
  $(BIN)/bh_omp_v3 $(BIN)/bh_omp_v3s

$(BIN)/bh_omp_v1: $(SRC)/BH_omp_v1.cpp $(SRC)/common.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "[OK] $@"

$(BIN)/bh_omp_v2: $(SRC)/BH_omp_v2.cpp $(SRC)/common.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "[OK] $@"

$(BIN)/bh_omp_v3: $(SRC)/BH_omp_v3.cpp $(SRC)/common.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "[OK] $@"

$(BIN)/bh_omp_v3s: $(SRC)/BH_omp_v3.cpp $(SRC)/common.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DSPATIAL_SORT -o $@ $<
	@echo "[OK] $@  (avec tri spatial)"

# ── Phase 2 : MPI + OMP ───────────────────────────────────────
mpi: dirs \
  $(BIN)/bh_mpi_v1 $(BIN)/bh_mpi_v2 $(BIN)/bh_mpi_v3

$(BIN)/bh_mpi_v1: $(SRC)/BH_mpi_v1.cpp $(SRC)/common.hpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIDEF) -o $@ $<
	@echo "[OK] $@"

$(BIN)/bh_mpi_v2: $(SRC)/BH_mpi_v2.cpp $(SRC)/common.hpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIDEF) -o $@ $<
	@echo "[OK] $@"

$(BIN)/bh_mpi_v3: $(SRC)/BH_mpi_v3.cpp $(SRC)/common.hpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIDEF) -o $@ $<
	@echo "[OK] $@"

clean:
	rm -rf $(BIN) results

benchmark: all
	bash scripts/benchmark.sh

benchmark_quick: all
	bash scripts/benchmark.sh --quick
