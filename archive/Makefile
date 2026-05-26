# ==============================================================
#  Makefile — Barnes-Hut N-body : étude de parallélisation
#
#  ARBRE DES VARIANTES
#  ────────────────────
#  Référence
#    bh_seq          séquentiel pur (BH_omp_v1 sans OMP)
#
#  Phase 1 — OpenMP (4 stratégies sur la boucle de forces)
#    bh_omp_v1       schedule(static)
#    bh_omp_v2       schedule(dynamic, 32)
#    bh_omp_v3       schedule(guided, 16)
#    bh_omp_v3s      schedule(guided, 16) + tri spatial initial
#
#  Phase 2 — MPI × OMP (3 architectures de communication)
#    bh_mpi_v1       bandes Y  + Allgather 7f  + Alltoallv
#    bh_mpi_v2       blocs idx + Allgather 7f  (pas de redistrib.)
#    bh_mpi_v3       bandes Y  + Allgather 3f  (communication légère)
#
#  Phase 3 — Version finale hybride
#    bh_hybrid       bandes Y + Allgather 3f + OMP dynamic(32)
#                    + fusion reset/force + buffers persistants
#                    + tri spatial périodique + mesure comm séparée
#
#  Phase 4 — CUDA
#    bh_cuda         arbre CPU → sérialisé SOA → kernel forces GPU
#                    (fallback CPU si nvcc absent)
# ==============================================================

CXX    := g++
MPICXX := mpicxx
NVCC   := nvcc

CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -I src
OMPFLAGS  := -fopenmp
MPIDEF    := -DUSE_MPI

CUDA_ARCH  := sm_90
NVCCFLAGS  := -O3 -std=c++17 -arch=$(CUDA_ARCH) -I src \
              --expt-relaxed-constexpr -Xcompiler -fopenmp

BIN := bin
SRC := src

NVCC_FOUND := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes || echo no)

.PHONY: all seq omp mpi hybrid cuda cuda_cpu clean benchmark benchmark_quick dirs

ifeq ($(NVCC_FOUND),yes)
all: dirs seq omp mpi hybrid cuda
else
all: dirs seq omp mpi hybrid cuda_cpu
	@echo ""
	@echo "[INFO] nvcc absent — bh_cuda compilé en fallback CPU"
	@echo "       Pour le vrai GPU : installer cuda-toolkit puis make cuda"
endif

dirs:
	@mkdir -p $(BIN) results

# ── Séquentiel (référence baseline) ──────────────────────────
seq: dirs $(BIN)/bh_seq
$(BIN)/bh_seq: $(SRC)/BH_omp_v1.cpp $(SRC)/common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "[OK] $@"

# ── Phase 1 : OpenMP ─────────────────────────────────────────
omp: dirs $(BIN)/bh_omp_v1 $(BIN)/bh_omp_v2 $(BIN)/bh_omp_v3 $(BIN)/bh_omp_v3s

$(BIN)/bh_omp_v1: $(SRC)/BH_omp_v1.cpp $(SRC)/common.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "[OK] $@  (static)"

$(BIN)/bh_omp_v2: $(SRC)/BH_omp_v2.cpp $(SRC)/common.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "[OK] $@  (dynamic 32)"

$(BIN)/bh_omp_v3: $(SRC)/BH_omp_v3.cpp $(SRC)/common.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "[OK] $@  (guided 16)"

$(BIN)/bh_omp_v3s: $(SRC)/BH_omp_v3.cpp $(SRC)/common.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DSPATIAL_SORT -o $@ $<
	@echo "[OK] $@  (guided 16 + tri spatial)"

# ── Phase 2 : MPI × OMP ──────────────────────────────────────
mpi: dirs $(BIN)/bh_mpi_v1 $(BIN)/bh_mpi_v2 $(BIN)/bh_mpi_v3

$(BIN)/bh_mpi_v1: $(SRC)/BH_mpi_v1.cpp $(SRC)/common.hpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIDEF) -o $@ $<
	@echo "[OK] $@  (bandes Y + Allgather 7f)"

$(BIN)/bh_mpi_v2: $(SRC)/BH_mpi_v2.cpp $(SRC)/common.hpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIDEF) -o $@ $<
	@echo "[OK] $@  (blocs idx + Allgather 7f)"

$(BIN)/bh_mpi_v3: $(SRC)/BH_mpi_v3.cpp $(SRC)/common.hpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIDEF) -o $@ $<
	@echo "[OK] $@  (bandes Y + Allgather 3f léger)"

# ── Phase 3 : Hybride final ───────────────────────────────────
hybrid: dirs $(BIN)/bh_hybrid
$(BIN)/bh_hybrid: $(SRC)/BH_hybrid.cpp $(SRC)/common.hpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIDEF) -o $@ $<
	@echo "[OK] $@  (MPI 3f + OMP dynamic32 + fusion + tri)"

# ── Phase 4 : CUDA — GPU réel ─────────────────────────────────
cuda: dirs
	$(NVCC) $(NVCCFLAGS) -o $(BIN)/bh_cuda $(SRC)/BH_cuda.cu
	@echo "[OK] $(BIN)/bh_cuda  (GPU $(CUDA_ARCH))"

# ── Phase 4 : CUDA — fallback CPU ────────────────────────────
cuda_cpu: dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DNO_CUDA -x c++ \
	    -o $(BIN)/bh_cuda $(SRC)/BH_cuda.cu
	@echo "[OK] $(BIN)/bh_cuda  (fallback CPU)"

clean:
	rm -rf $(BIN) results

benchmark: all
	bash scripts/benchmark.sh

benchmark_quick: all
	bash scripts/benchmark.sh --quick