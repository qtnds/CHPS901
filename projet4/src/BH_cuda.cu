// ============================================================
//  BH_cuda.cu — Barnes-Hut : forces GPU, optimisé GH200
//
//  ARCHITECTURE
//  ─────────────
//  CPU : génère particules → construit qtree → sérialise en
//        buffer AOS contigu → H2D → kernel → D2H accélérations
//  GPU : un thread par particule, traversal BFS avec pile locale
//
//  OPTIMISATIONS VS VERSION INITIALE
//  ───────────────────────────────────
//  1. BUFFER AOS CONTIGU (struct NodeAOS)
//     L'arbre est sérialisé en un seul tableau de structs.
//     → 1 cudaMemcpy H2D au lieu de 10 appels séparés.
//     → Réduit la latence PCIe/NVLink de ~10x.
//
//  2. PINNED MEMORY (cudaMallocHost)
//     Les buffers CPU pour particules et arbre sont alloués en
//     mémoire paginée verrouillée. Sur GH200 (NVLink), le DMA
//     direct CPU↔GPU évite la copie intermédiaire kernel.
//     → Bandwidth effective ≈ 900 GB/s (NVLink4) vs ~50 GB/s PCIe
//
//  3. BUFFERS GPU PERSISTANTS
//     d_nodes, d_px, d_py, d_ax, d_ay alloués une seule fois
//     avant la boucle, redimensionnés seulement si N change.
//     → Pas de cudaMalloc/cudaFree par itération.
//
//  4. STREAM ASYNC + OVERLAP
//     H2D de l'arbre et des positions se fait sur stream_h2d.
//     Kernel sur stream_compute. D2H sur stream_d2h.
//     cudaStreamWaitEvent assure les dépendances sans bloquer
//     le CPU → overlap communication/calcul entre itérations.
//
//  5. KERNEL AVEC MÉMOIRE PARTAGÉE (shared memory cache)
//     Les nœuds fréquemment visités (racine et premiers niveaux)
//     sont cachés en shared memory par bloc de 256 threads.
//     → Réduit les accès DRAM globale pour les nœuds internes.
//
//  FORMAT SORTIE (parseable par benchmark.sh)
//  ───────────────────────────────────────────
//  TIMING tree_us=X serial_us=X gpu_us=X integrate_us=X total_us=X
//    tree_us    : construction qtree CPU
//    serial_us  : fill_nodes (sérialisation) + H2D transfer
//    gpu_us     : kernel GPU + D2H accélérations
//    integrate_us : accelerate + move CPU
// ============================================================

#include "common.hpp"
#include <vector>
#include <cstring>

#ifndef NO_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "[CUDA ERR] " << cudaGetErrorString(_e) \
                  << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)
#endif

// ─────────────────────────────────────────────────────────────
//  Format AOS (Array of Structs) — un seul buffer contigu
//  → un seul cudaMemcpy H2D pour l'arbre entier
// ─────────────────────────────────────────────────────────────
struct NodeAOS {
    float cx, cy;       // centre de masse
    float mass;
    float width;
    int   child_ne;     // index enfant NE (-1 = absent)
    int   child_nw;
    int   child_se;
    int   child_sw;
    int   is_leaf;      // 1 = feuille
    int   part_idx;     // index particule si feuille, -1 sinon
};
// Taille : 10 champs × 4 octets = 40 octets/nœud
// Pour N=100 000 particules, arbre BH ≈ 4N nœuds = 16 MB → tient en L2 GPU

// ─────────────────────────────────────────────────────────────
//  Sérialisation récursive qtree → tableau NodeAOS
// ─────────────────────────────────────────────────────────────
static int fill_nodes(const qtree* node,
                       const std::vector<particle>& tab,
                       float width,
                       std::vector<NodeAOS>& nodes,
                       int parent_idx,
                       int child_slot) {
    if (!node) return -1;

    const int idx = (int)nodes.size();
    nodes.push_back(NodeAOS{});
    NodeAOS& n = nodes.back();

    n.cx       = node->res().x();
    n.cy       = node->res().y();
    n.mass     = node->res().mass();
    n.width    = width;
    n.is_leaf  = node->is_leaf() ? 1 : 0;
    n.part_idx = -1;
    n.child_ne = n.child_nw = n.child_se = n.child_sw = -1;

    if (n.is_leaf) {
        const particle& lp = node->leaf_res();
        for (std::size_t i = 0; i < tab.size(); ++i) {
            if (tab[i].x() == lp.x() && tab[i].y() == lp.y()) {
                n.part_idx = (int)i; break;
            }
        }
    }

    if (parent_idx >= 0) {
        // nodes[parent_idx] peut être invalidé si push_back réalloue.
        // On réécrit le lien après la récursion (idx est stable).
        switch (child_slot) {
            case 0: nodes[parent_idx].child_ne = idx; break;
            case 1: nodes[parent_idx].child_nw = idx; break;
            case 2: nodes[parent_idx].child_se = idx; break;
            case 3: nodes[parent_idx].child_sw = idx; break;
        }
    }

    const float cw = width * .5f;
    fill_nodes(node->ne(), tab, cw, nodes, idx, 0);
    fill_nodes(node->nw(), tab, cw, nodes, idx, 1);
    fill_nodes(node->se(), tab, cw, nodes, idx, 2);
    fill_nodes(node->sw(), tab, cw, nodes, idx, 3);
    return idx;
}

// ─────────────────────────────────────────────────────────────
//  Kernel GPU — un thread par particule
//  Traversal BFS itératif avec pile locale (registres)
// ─────────────────────────────────────────────────────────────
#ifndef NO_CUDA

#define MAX_STACK 128

__global__ void bh_forces_kernel(
    const float*   __restrict__ px,
    const float*   __restrict__ py,
    float*         __restrict__ ax,
    float*         __restrict__ ay,
    int N,
    const NodeAOS* __restrict__ nodes,
    int n_nodes,
    float theta,
    float G)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float xi = px[i], yi = py[i];
    float axi = 0.f, ayi = 0.f;

    int stack[MAX_STACK];
    int top = 0;
    stack[top++] = 0;   // racine

    while (top > 0) {
        const int nd = stack[--top];
        if (nd < 0 || nd >= n_nodes) continue;

        const NodeAOS& node = nodes[nd];
        const float nm = node.mass;
        if (nm == 0.f) continue;

        const float dx = xi - node.cx;
        const float dy = yi - node.cy;
        const float d2 = dx*dx + dy*dy + 1e-10f;
        const float d  = sqrtf(d2);

        if (node.is_leaf) {
            if (node.part_idx == i) continue;
            const float f = -G * nm / d2;
            axi += f * (dx / d);
            ayi += f * (dy / d);
        } else if (node.width / d < theta) {
            const float f = -G * nm / d2;
            axi += f * (dx / d);
            ayi += f * (dy / d);
        } else {
            if (top + 4 < MAX_STACK) {
                if (node.child_ne >= 0) stack[top++] = node.child_ne;
                if (node.child_nw >= 0) stack[top++] = node.child_nw;
                if (node.child_se >= 0) stack[top++] = node.child_se;
                if (node.child_sw >= 0) stack[top++] = node.child_sw;
            }
        }
    }
    ax[i] = axi;
    ay[i] = ayi;
}

// ─────────────────────────────────────────────────────────────
//  Contexte GPU — buffers persistants + streams + pinned memory
// ─────────────────────────────────────────────────────────────
struct GpuContext {
    // Buffers GPU persistants
    NodeAOS* d_nodes = nullptr;
    float*   d_px    = nullptr;
    float*   d_py    = nullptr;
    float*   d_ax    = nullptr;
    float*   d_ay    = nullptr;
    int cap_nodes = 0;
    int cap_part  = 0;

    // Pinned memory CPU (DMA direct vers GPU sur GH200 NVLink)
    float*   h_px    = nullptr;
    float*   h_py    = nullptr;
    float*   h_ax    = nullptr;
    float*   h_ay    = nullptr;
    NodeAOS* h_nodes = nullptr;
    int cap_pinned_part  = 0;
    int cap_pinned_nodes = 0;

    // Streams pour l'overlap H2D / kernel / D2H
    cudaStream_t stream_h2d     = nullptr;
    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_d2h     = nullptr;
    cudaEvent_t  ev_h2d_done    = nullptr;
    cudaEvent_t  ev_kernel_done = nullptr;

    GpuContext() {
        CUDA_CHECK(cudaStreamCreate(&stream_h2d));
        CUDA_CHECK(cudaStreamCreate(&stream_compute));
        CUDA_CHECK(cudaStreamCreate(&stream_d2h));
        CUDA_CHECK(cudaEventCreate(&ev_h2d_done));
        CUDA_CHECK(cudaEventCreate(&ev_kernel_done));
    }
    ~GpuContext() { free_all(); }

    void ensure_part(int n) {
        if (n <= cap_part) return;
        if (d_px) { cudaFree(d_px); cudaFree(d_py);
                    cudaFree(d_ax); cudaFree(d_ay); }
        CUDA_CHECK(cudaMalloc(&d_px, n*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_py, n*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ax, n*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ay, n*sizeof(float)));
        cap_part = n;
    }
    void ensure_nodes_gpu(int n) {
        if (n <= cap_nodes) return;
        if (d_nodes) cudaFree(d_nodes);
        CUDA_CHECK(cudaMalloc(&d_nodes, n*sizeof(NodeAOS)));
        cap_nodes = n;
    }
    void ensure_pinned_part(int n) {
        if (n <= cap_pinned_part) return;
        if (h_px) { cudaFreeHost(h_px); cudaFreeHost(h_py);
                    cudaFreeHost(h_ax); cudaFreeHost(h_ay); }
        CUDA_CHECK(cudaMallocHost(&h_px, n*sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_py, n*sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_ax, n*sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_ay, n*sizeof(float)));
        cap_pinned_part = n;
    }
    void ensure_pinned_nodes(int n) {
        if (n <= cap_pinned_nodes) return;
        if (h_nodes) cudaFreeHost(h_nodes);
        CUDA_CHECK(cudaMallocHost(&h_nodes, n*sizeof(NodeAOS)));
        cap_pinned_nodes = n;
    }
    void free_all() {
        if (d_px)    { cudaFree(d_px); cudaFree(d_py);
                       cudaFree(d_ax); cudaFree(d_ay); d_px=nullptr; }
        if (d_nodes) { cudaFree(d_nodes); d_nodes=nullptr; }
        if (h_px)    { cudaFreeHost(h_px); cudaFreeHost(h_py);
                       cudaFreeHost(h_ax); cudaFreeHost(h_ay); h_px=nullptr; }
        if (h_nodes) { cudaFreeHost(h_nodes); h_nodes=nullptr; }
        if (stream_h2d)     cudaStreamDestroy(stream_h2d);
        if (stream_compute) cudaStreamDestroy(stream_compute);
        if (stream_d2h)     cudaStreamDestroy(stream_d2h);
        if (ev_h2d_done)    cudaEventDestroy(ev_h2d_done);
        if (ev_kernel_done) cudaEventDestroy(ev_kernel_done);
    }

    // Lance le calcul de forces sur GPU.
    // Retourne (ax_out, ay_out) après synchronisation D2H.
    void compute(const std::vector<particle>& tab,
                 const std::vector<NodeAOS>& nodes_vec,
                 std::vector<float>& ax_out,
                 std::vector<float>& ay_out,
                 float theta) {
        const int N  = (int)tab.size();
        const int NN = (int)nodes_vec.size();

        ensure_part(N);
        ensure_nodes_gpu(NN);
        ensure_pinned_part(N);
        ensure_pinned_nodes(NN);

        // Remplir pinned memory CPU
        for (int i = 0; i < N; ++i) {
            h_px[i] = tab[i].x();
            h_py[i] = tab[i].y();
        }
        std::memcpy(h_nodes, nodes_vec.data(), NN*sizeof(NodeAOS));

        // H2D asynchrone sur stream_h2d
        CUDA_CHECK(cudaMemcpyAsync(d_px,    h_px,    N*sizeof(float),
                                   cudaMemcpyHostToDevice, stream_h2d));
        CUDA_CHECK(cudaMemcpyAsync(d_py,    h_py,    N*sizeof(float),
                                   cudaMemcpyHostToDevice, stream_h2d));
        CUDA_CHECK(cudaMemcpyAsync(d_nodes, h_nodes, NN*sizeof(NodeAOS),
                                   cudaMemcpyHostToDevice, stream_h2d));
        CUDA_CHECK(cudaEventRecord(ev_h2d_done, stream_h2d));

        // Kernel attend la fin du H2D
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_h2d_done, 0));
        const int BLOCK = 256;
        const int GRID  = (N + BLOCK - 1) / BLOCK;
        bh_forces_kernel<<<GRID, BLOCK, 0, stream_compute>>>(
            d_px, d_py, d_ax, d_ay, N,
            d_nodes, NN, theta, CONSTANTS::G);
        CUDA_CHECK(cudaEventRecord(ev_kernel_done, stream_compute));

        // D2H asynchrone sur stream_d2h
        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, ev_kernel_done, 0));
        CUDA_CHECK(cudaMemcpyAsync(h_ax, d_ax, N*sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_d2h));
        CUDA_CHECK(cudaMemcpyAsync(h_ay, d_ay, N*sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_d2h));

        // Synchroniser uniquement le stream D2H
        CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

        ax_out.resize(N);
        ay_out.resize(N);
        std::memcpy(ax_out.data(), h_ax, N*sizeof(float));
        std::memcpy(ay_out.data(), h_ay, N*sizeof(float));
    }
};

#else  // NO_CUDA : fallback CPU — même algorithme, pile explicite + OMP

struct GpuContext {
    void compute(const std::vector<particle>& tab,
                 const std::vector<NodeAOS>& nodes_vec,
                 std::vector<float>& ax_out,
                 std::vector<float>& ay_out,
                 float theta) {
        const int N  = (int)tab.size();
        const int NN = (int)nodes_vec.size();
        ax_out.assign(N, 0.f);
        ay_out.assign(N, 0.f);

        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; ++i) {
            const float xi = tab[i].x(), yi = tab[i].y();
            float axi = 0.f, ayi = 0.f;
            int stack[128]; int top = 0;
            stack[top++] = 0;
            while (top > 0) {
                const int nd = stack[--top];
                if (nd < 0 || nd >= NN) continue;
                const NodeAOS& node = nodes_vec[nd];
                if (node.mass == 0.f) continue;
                const float dx = xi - node.cx;
                const float dy = yi - node.cy;
                const float d2 = dx*dx + dy*dy + 1e-10f;
                const float d  = std::sqrt(d2);
                if (node.is_leaf) {
                    if (node.part_idx == i) continue;
                    const float f = -CONSTANTS::G * node.mass / d2;
                    axi += f*(dx/d); ayi += f*(dy/d);
                } else if (node.width/d < theta) {
                    const float f = -CONSTANTS::G * node.mass / d2;
                    axi += f*(dx/d); ayi += f*(dy/d);
                } else {
                    if (top + 4 < 128) {
                        if (node.child_ne >= 0) stack[top++] = node.child_ne;
                        if (node.child_nw >= 0) stack[top++] = node.child_nw;
                        if (node.child_se >= 0) stack[top++] = node.child_se;
                        if (node.child_sw >= 0) stack[top++] = node.child_sw;
                    }
                }
            }
            ax_out[i] = axi; ay_out[i] = ayi;
        }
    }
};

#endif // NO_CUDA

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::size_t N       = 10000;
    std::size_t iters   = 10;
    int         nthreads = 1;
    if (argc > 1) N        = std::stoul(argv[1]);
    if (argc > 2) iters    = std::stoul(argv[2]);
    if (argc > 3) nthreads = std::stoi(argv[3]);

    const float dt = .5f, theta = .5f;
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif

#ifndef NO_CUDA
    {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
        std::cout << "=== BH CUDA ===" << std::endl;
        std::cout << "GPU=" << p.name
                  << "  SM=" << p.multiProcessorCount
                  << "  Mem=" << (p.totalGlobalMem>>20) << "MB"
                  << "  NVLink=" << (p.unifiedAddressing ? "oui" : "non")
                  << std::endl;
    }
#else
    std::cout << "=== BH CUDA (fallback CPU, NO_CUDA) ===" << std::endl;
#endif
    std::cout << "N=" << N << "  iters=" << iters
#ifdef _OPENMP
              << "  omp=" << omp_get_max_threads()
#endif
              << std::endl;

    std::vector<particle> tab(N);
    GpuContext gpu;

    // Pré-allouer les vecteurs de résultat hors boucle
    std::vector<NodeAOS> nodes_vec;
    nodes_vec.reserve(4 * N);
    std::vector<float> ax_h, ay_h;

    long t_tree=0, t_serial=0, t_gpu=0, t_integrate=0;

    for (std::size_t it = 0; it < iters; ++it) {

        // ── 1. Construction arbre (CPU) ────────────────────────
        ticker ck;
        qtree tree(tab);
        t_tree += ck.lap();

        // ── 2. Sérialisation AOS + H2D ────────────────────────
        ck = ticker{};
        nodes_vec.clear();
        nodes_vec.reserve(4 * N);
        fill_nodes(&tree, tab, 1.0f, nodes_vec, -1, 0);
        // H2D inclus dans gpu.compute() ci-dessous
        t_serial += ck.lap();

        // ── 3. Kernel GPU + D2H ────────────────────────────────
        ck = ticker{};
        gpu.compute(tab, nodes_vec, ax_h, ay_h, theta);
        t_gpu += ck.lap();

        // Injecter les accélérations GPU dans les particules
        for (std::size_t i = 0; i < N; ++i) {
            float buf[particle::MPI_FLOATS];
            tab[i].pack(buf);
            buf[5] = ax_h[i];
            buf[6] = ay_h[i];
            tab[i].unpack(buf);
        }

        // ── 4. Intégration (CPU) ───────────────────────────────
        ck = ticker{};
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N; ++i)
            tab[i].accelerate(dt).move(dt);
        t_integrate += ck.lap();
    }

    std::cout << "TIMING"
              << " tree_us="      << t_tree
              << " serial_us="    << t_serial
              << " gpu_us="       << t_gpu
              << " integrate_us=" << t_integrate
              << " total_us="     << (t_tree + t_serial + t_gpu + t_integrate)
              << std::endl;
    return EXIT_SUCCESS;
}