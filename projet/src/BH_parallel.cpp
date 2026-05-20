// ============================================================
//  Barnes-Hut N-body simulation — OpenMP + MPI version
//
//  Stratégie parallèle :
//
//  [OpenMP]
//    - Boucle de calcul d'interaction : chaque thread prend
//      en charge un sous-ensemble de particules (parallel for,
//      schedule dynamic pour l'équilibrage de charge car le
//      nombre d'opérations BH par particule varie).
//    - Boucle d'intégration (accelerate + move) : embarrassingly
//      parallel, schedule static suffit.
//    - Construction du qtree : NON parallélisée (insertions
//      séquentielles requises, la structure arborescente n'est
//      pas thread-safe). Le coût est O(N log N), négligeable
//      devant le calcul de forces O(N log N theta).
//
//  [MPI]
//    - Décomposition en domaines : l'espace [-0.5, 0.5]^2 est
//      découpé en bandes horizontales égales (une par rang).
//      Chaque rang possède les particules dont la coordonnée y
//      tombe dans sa bande.
//    - À chaque itération :
//        1. Chaque rang construit son qtree LOCAL sur ses
//           particules.
//        2. Allgather des particules de tous les rangs →
//           chaque rang dispose de l'ensemble complet pour
//           calculer l'arbre GLOBAL utilisé dans calc_interaction.
//           (Alternative : envoyer seulement les résumés de
//           centre-de-masse — optimisation future § amélioration.)
//        3. Calcul de forces sur les particules LOCALES en
//           interrogeant l'arbre global.
//        4. Intégration locale.
//        5. Redistribution : les particules qui ont quitté la
//           bande locale sont renvoyées au bon rang (MPI_Alltoallv).
// ============================================================

#include <iostream>
#include <iomanip>
#include <ios>
#include <cstdlib>
#include <random>
#include <array>
#include <cfloat>
#include <stdexcept>
#include <sys/time.h>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cstring>

// ── OpenMP ──────────────────────────────────────────────────
#ifdef _OPENMP
#include <omp.h>
#endif

// ── MPI ─────────────────────────────────────────────────────
#ifdef USE_MPI
#include <mpi.h>
#endif

namespace CONSTANTS {
    constexpr float G = 6.6742e-11f;
}

// ============================================================
//  Point2D / Vector2D
// ============================================================

struct Point2D {
    private:
        struct print_coord {
            float coord;
            friend std::ostream& operator<<(std::ostream& os, const print_coord& c) {
                const auto dp{std::cout.precision()};
                return os << std::right << std::setw(8) << std::setprecision(5)
                          << c.coord << std::setw(0) << std::setprecision(dp);
            }
        };
    public:
        float x = 0.f, y = 0.f;
        friend std::ostream& operator<<(std::ostream& os, const Point2D& pt) {
            return os << '(' << print_coord{pt.x} << ',' << print_coord{pt.y} << ')';
        }
        Point2D& operator+=(const Point2D& v) { x += v.x; y += v.y; return *this; }
        friend Point2D operator+(const Point2D& a, const Point2D& b) { return {a.x+b.x, a.y+b.y}; }
        friend Point2D&& operator+(const Point2D& a, Point2D&& b)    { b += a; return std::move(b); }
        friend Point2D&& operator+(Point2D&& a, const Point2D& b)    { return b + std::move(a); }
        friend Point2D&& operator+(Point2D&& a, Point2D&& b)         { return a + std::move(b); }
        Point2D operator-(const Point2D& p) const { return {x-p.x, y-p.y}; }
        bool operator<(const Point2D& p) const { return (x < p.x || y < p.y); }
};

struct Vector2D : Point2D {
    Vector2D() = default;
    Vector2D(float x, float y) : Point2D({x,y}) {}
    Vector2D(const Point2D& p1, const Point2D& p2) : Point2D(p1-p2) {}
    Vector2D(const Vector2D&) = default;
    Vector2D(Vector2D&&) = default;
    Vector2D& operator=(const Vector2D&) = default;
    Vector2D& operator=(Vector2D&&) = default;
    float magnitude() const { return std::hypotf(x,y); }
    Vector2D unit() const { return Vector2D(x/magnitude(), y/magnitude()); }
    Vector2D operator*(float a) const { return Vector2D(x*a, y*a); }
    friend Vector2D operator*(float a, const Vector2D& v) { return v*a; }
};

// ============================================================
//  particle
// ============================================================

class particle {
    Point2D   pos_;
    Vector2D  speed_;
    float     mass_ = 0.f;
    Vector2D  acceleration_;

    struct print_mass {
        float mass;
        friend std::ostream& operator<<(std::ostream& os, const print_mass& c) {
            const auto dp{std::cout.precision()};
            return os << std::left << std::setw(2) << std::setprecision(5)
                      << c.mass << std::setw(0) << std::setprecision(dp);
        }
    };

    // ── RNG privé ─────────────────────────────────────────────
    class MyRNG {
        static std::mt19937 generator;
        static float get_normal_float() {
            static std::normal_distribution<float> d(0.f, 1.f);
            return d(generator);
        }
        static float get_uni_float() {
            std::uniform_real_distribution<float> d(-.5f, std::nextafter(.5f, FLT_MAX));
            return d(generator);
        }
    public:
        static float get_mass()  { return std::abs(get_normal_float()); }
        static float get_coord() { return get_uni_float(); }
        static float get_speed() { return get_normal_float() * .1f; }
    };

public:
    // ── Constructeurs ─────────────────────────────────────────
    particle()
        : pos_({MyRNG::get_coord(), MyRNG::get_coord()})
        , speed_({MyRNG::get_speed(), MyRNG::get_speed()})
        , mass_(MyRNG::get_mass()) {}

    particle(const particle&) = default;
    particle(particle&&) = default;
    particle& operator=(const particle&) = default;
    particle& operator=(particle&&) = default;
    explicit particle(const Point2D& p) : pos_(p) {}

    // ── Accesseurs ────────────────────────────────────────────
    const float& x()    const { return pos_.x; }
    const float& y()    const { return pos_.y; }
    const float& mass() const { return mass_; }
    bool has_mass()     const { return 0.f < mass_; }

    // ── Sérialisation MPI (POD flat layout) ───────────────────
    //   [pos.x, pos.y, speed.x, speed.y, mass, acc.x, acc.y]
    static constexpr int MPI_FLOATS = 7;

    void pack(float* buf) const {
        buf[0] = pos_.x;   buf[1] = pos_.y;
        buf[2] = speed_.x; buf[3] = speed_.y;
        buf[4] = mass_;
        buf[5] = acceleration_.x; buf[6] = acceleration_.y;
    }
    void unpack(const float* buf) {
        pos_.x = buf[0];   pos_.y = buf[1];
        speed_.x = buf[2]; speed_.y = buf[3];
        mass_    = buf[4];
        acceleration_.x = buf[5]; acceleration_.y = buf[6];
    }

    // ── Physique ──────────────────────────────────────────────
    particle& operator+=(const particle& p) {
        const float nm = mass_ + p.mass_;
        pos_.x = (pos_.x * mass_ + p.pos_.x * p.mass_) / nm;
        pos_.y = (pos_.y * mass_ + p.pos_.y * p.mass_) / nm;
        mass_  = nm;
        return *this;
    }

    particle& apply_force(const particle& p) {
        Vector2D res(pos_, p.pos_);
        const float mag = res.magnitude();
        const float acc = -1.f * CONSTANTS::G * p.mass_ / std::pow(mag, 2.f);
        acceleration_ += res.unit() * acc;
        return *this;
    }

    particle& accelerate(float dt) {
        speed_ += acceleration_ * dt;
        return *this;
    }

    particle& move(float dt) {
        pos_ += speed_ * dt;
        check_and_update_with_rebounds(pos_.x, speed_.x, .5f);
        check_and_update_with_rebounds(pos_.y, speed_.y, .5f);
        return *this;
    }

    void reset_acceleration() { acceleration_ = Vector2D(0.f, 0.f); }

    bool operator<(const particle& p) const { return pos_ < p.pos_; }

    friend std::ostream& operator<<(std::ostream& os, const particle& p) {
        return os << "{ " << p.pos_ << ": " << print_mass{p.mass_} << " }";
    }

private:
    static long long get_bounces(float val, float limit) {
        return 1.f + val / (limit * 2.f);
    }
    static float maybe_invert(float val, long long n) {
        return n & 1 ? -1.f * val : val;
    }
    static float get_rebound_pos(float val, float limit, long long b) {
        return limit - std::fmod(val + 2.f * ((b >> 1) << 1), limit);
    }
    static void update_with_rebounds(float& pos, float& speed, float limit) {
        const long long b = get_bounces(pos, limit);
        speed = maybe_invert(speed, b);
        pos   = get_rebound_pos(pos, limit, b);
    }
    static void check_and_update_with_rebounds(float& pos, float& speed, float limit) {
        if (pos < -limit) update_with_rebounds(pos, speed, -limit);
        if (pos >  limit) update_with_rebounds(pos, speed,  limit);
    }
};
std::mt19937 particle::MyRNG::generator(std::random_device{}());

// ============================================================
//  qtree  (inchangé hormis le friend calc_interaction)
// ============================================================

class qtree {
    qtree *ne_=nullptr, *nw_=nullptr, *se_=nullptr, *sw_=nullptr;
    Point2D pos_ = {0.f, 0.f};
    union { const particle* leaf_res_; particle* res_; };
    float width_=1.f, height_=1.f;

    qtree(const particle* part, const Point2D& p, float w, float h)
        : pos_(p), leaf_res_(part)
        , width_ (0.f!=w ? w : throw std::domain_error("null width"))
        , height_(0.f!=h ? h : throw std::domain_error("null height")) {}

public:
    template <class T>
    explicit qtree(const T& container) : res_(new particle(pos_)) {
        for (auto it = container.cbegin(); it != container.cend(); ++it)
            add_particle(&(*it));
    }

    ~qtree() {
        if (!is_leaf()) delete res_;
        if (ne_) delete ne_;
        if (nw_) delete nw_;
        if (se_) delete se_;
        if (sw_) delete sw_;
    }

    void add_particle(const particle* p) {
        if (is_leaf() && leaf_res_->has_mass()) {
            insert_particle(leaf_res_);
            res_ = new particle(*leaf_res_);
        }
        *res_ += *p;
        insert_particle(p);
    }

    bool has_particle() const { return res_->has_mass(); }
    bool is_leaf()      const { return !(ne_||nw_||se_||sw_); }
    const particle& leaf_res() const { return *leaf_res_; }
    const particle& res()      const { return *res_; }

    friend std::ostream& operator<<(std::ostream& os, const qtree& t) {
        return qtree::print(os, t, 0);
    }

private:
    struct indent {
        std::size_t l;
        friend std::ostream& operator<<(std::ostream& os, const indent& i) {
            for (std::size_t j=0; j<i.l; ++j) os << "|\t";
            return os;
        }
    };
    static std::ostream& print(std::ostream& os, const qtree* t, std::size_t l) {
        return (nullptr==t) ? (os<<"nil") : print(os,*t,l);
    }
    static std::ostream& print(std::ostream& os, const qtree& t, std::size_t l) {
        os << "{ " << *t.res_ << ' ';
        if (!t.is_leaf()) {
            os << '\n' << indent{l+1} << "ne:"; print(os,t.ne_,l+1);
            os << '\n' << indent{l+1} << "nw:"; print(os,t.nw_,l+1);
            os << '\n' << indent{l+1} << "se:"; print(os,t.se_,l+1);
            os << '\n' << indent{l+1} << "sw:"; print(os,t.sw_,l+1);
            os << '\n' << indent{l};
        }
        os << '}';
        return os;
    }

    void insert_particle(const particle* p) {
        const float nw = width_  * .5f;
        const float nh = height_ * .5f;
        const Point2D c = {nw*.5f, nh*.5f};
        if      (p->x()<pos_.x && p->y()<pos_.y) {
            if (!sw_) sw_=new qtree(p,pos_+Vector2D{-c.x,-c.y},nw,nh); else sw_->add_particle(p);
        } else if (p->x()<pos_.x && p->y()>=pos_.y) {
            if (!nw_) nw_=new qtree(p,pos_+Vector2D{-c.x, c.y},nw,nh); else nw_->add_particle(p);
        } else if (p->x()>=pos_.x && p->y()<pos_.y) {
            if (!se_) se_=new qtree(p,pos_+Vector2D{ c.x,-c.y},nw,nh); else se_->add_particle(p);
        } else if (p->x()>=pos_.x && p->y()>=pos_.y) {
            if (!ne_) ne_=new qtree(p,pos_+Vector2D{ c.x, c.y},nw,nh); else ne_->add_particle(p);
        }
    }

public:
    friend void calc_interaction(particle& p, const qtree* tree, float theta) {
        if (!tree) return;
        if (tree->is_leaf() && tree->leaf_res_ != &p) {
            p.apply_force(tree->leaf_res());
        } else {
            const float s = tree->width_;
            const float d = std::hypotf(tree->res_->x()-p.x(), tree->res_->y()-p.y());
            if (s/d > theta) {
                calc_interaction(p, tree->nw_, theta);
                calc_interaction(p, tree->sw_, theta);
                calc_interaction(p, tree->ne_, theta);
                calc_interaction(p, tree->se_, theta);
            } else
                p.apply_force(tree->res());
        }
    }
};

// ============================================================
//  timer
// ============================================================

struct timer {
    struct timeval start_, end_;
    std::string label_;
    explicit timer(const std::string& lbl="") : label_(lbl) { gettimeofday(&start_, nullptr); }
    ~timer() {
        gettimeofday(&end_, nullptr);
        const long us = (end_.tv_sec - start_.tv_sec)*1000000L
                      + (end_.tv_usec - start_.tv_usec);
        std::cout << label_ << us << " us" << std::endl;
    }
    long elapsed_us() const {
        struct timeval now; gettimeofday(&now, nullptr);
        return (now.tv_sec-start_.tv_sec)*1000000L + (now.tv_usec-start_.tv_usec);
    }
};

// ============================================================
//  Helpers
// ============================================================

template <class C1, class C2>
float get_cumulative_error(const C1& v1, const C2& v2) {
    float err = 0.f;
    auto it1=v1.cbegin(); auto it2=v2.cbegin();
    for (; it1!=v1.cend() && it2!=v2.cend(); ++it1,++it2)
        err += std::pow(it1->x()-it2->x(),2.f) + std::pow(it1->y()-it2->y(),2.f);
    return err;
}

// ============================================================
//  MPI helpers : sérialisation / redistribution
// ============================================================

#ifdef USE_MPI

// Sérialise un vecteur de particules → buffer float plat
static void particles_to_buf(const std::vector<particle>& v, std::vector<float>& buf) {
    buf.resize(v.size() * particle::MPI_FLOATS);
    for (std::size_t i=0; i<v.size(); ++i)
        v[i].pack(&buf[i * particle::MPI_FLOATS]);
}

// Désérialise un buffer float → vecteur de particules
static void buf_to_particles(const std::vector<float>& buf, std::size_t n,
                              std::vector<particle>& v) {
    v.resize(n);
    for (std::size_t i=0; i<n; ++i) {
        particle p(Point2D{});          // construit un dummy
        p.unpack(&buf[i * particle::MPI_FLOATS]);
        v[i] = std::move(p);
    }
}

// Retourne [y_min, y_max) de la bande du rang mpi_rank parmi mpi_size rangs.
// L'espace y est [-0.5, 0.5].
static void band_limits(int rank, int size, float& ymin, float& ymax) {
    const float band = 1.f / static_cast<float>(size);
    ymin = -0.5f + rank * band;
    ymax = ymin + band;
}

// Détermine à quel rang appartient une particule selon sa coordonnée y.
static int owner_rank(float y, int size) {
    int r = static_cast<int>((y + 0.5f) * size);
    if (r < 0)    r = 0;
    if (r >= size) r = size-1;
    return r;
}

// Allgather : réunit toutes les particules de tous les rangs sur chaque rang.
// Retourne le vecteur global (lecture seule pour la construction du qtree).
static void allgather_particles(const std::vector<particle>& local,
                                 std::vector<particle>& global,
                                 int mpi_size) {
    // 1. Taille locale
    int local_n = static_cast<int>(local.size());
    std::vector<int> all_n(mpi_size);
    MPI_Allgather(&local_n, 1, MPI_INT, all_n.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 2. Sérialiser
    std::vector<float> lbuf;
    particles_to_buf(local, lbuf);

    // 3. Allgatherv
    std::vector<int> displs(mpi_size, 0);
    std::vector<int> counts(mpi_size);
    for (int i=0; i<mpi_size; ++i) counts[i] = all_n[i] * particle::MPI_FLOATS;
    for (int i=1; i<mpi_size; ++i) displs[i] = displs[i-1] + counts[i-1];
    const int total_floats = displs[mpi_size-1] + counts[mpi_size-1];

    std::vector<float> gbuf(total_floats);
    MPI_Allgatherv(lbuf.data(), static_cast<int>(lbuf.size()), MPI_FLOAT,
                   gbuf.data(), counts.data(), displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);

    // 4. Désérialiser
    const int total_n = total_floats / particle::MPI_FLOATS;
    buf_to_particles(gbuf, total_n, global);
}

// Redistribution après move : envoie les particules qui ont migré
// vers un autre rang (Alltoallv).
static void redistribute_particles(std::vector<particle>& local,
                                    int mpi_rank, int mpi_size) {
    // Trier les particules à envoyer par rang destinataire
    std::vector<std::vector<particle>> to_send(mpi_size);
    for (auto& p : local) {
        const int dest = owner_rank(p.y(), mpi_size);
        to_send[dest].push_back(p);
    }

    // Sérialiser par destination
    std::vector<float> sbuf;
    std::vector<int> scounts(mpi_size, 0);
    for (int r=0; r<mpi_size; ++r) {
        scounts[r] = static_cast<int>(to_send[r].size()) * particle::MPI_FLOATS;
        std::vector<float> tmp;
        particles_to_buf(to_send[r], tmp);
        sbuf.insert(sbuf.end(), tmp.begin(), tmp.end());
    }

    // Échanger les tailles
    std::vector<int> rcounts(mpi_size);
    MPI_Alltoall(scounts.data(), 1, MPI_INT, rcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculer displs
    std::vector<int> sdispls(mpi_size,0), rdispls(mpi_size,0);
    for (int i=1; i<mpi_size; ++i) {
        sdispls[i] = sdispls[i-1]+scounts[i-1];
        rdispls[i] = rdispls[i-1]+rcounts[i-1];
    }
    const int total_recv = rdispls[mpi_size-1] + rcounts[mpi_size-1];

    std::vector<float> rbuf(total_recv);
    MPI_Alltoallv(sbuf.data(),  scounts.data(), sdispls.data(), MPI_FLOAT,
                  rbuf.data(),  rcounts.data(), rdispls.data(), MPI_FLOAT,
                  MPI_COMM_WORLD);

    buf_to_particles(rbuf, total_recv / particle::MPI_FLOATS, local);
}
#endif // USE_MPI

// ============================================================
//  main
// ============================================================

int main(int argc, char** argv) {

    // ── Init MPI ────────────────────────────────────────────
    int mpi_rank = 0, mpi_size = 1;
    (void)mpi_size;
#ifdef USE_MPI
    // MPI_THREAD_FUNNELED : le thread maître peut appeler MPI,
    // les threads OpenMP ne font que du calcul pur.
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "[WARN] MPI ne supporte pas MPI_THREAD_FUNNELED ("
                  << provided << "); comportement indéfini possible.\n";
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

    // ── Paramètres ──────────────────────────────────────────
    std::size_t n_particles = 10000;
    std::size_t iters       = 10;
    int         n_threads   = 1;
    (void)n_threads;
    const float dt          = .5f;
    const float theta       = .5f;

    if (argc > 1) n_particles = std::stoul(argv[1]);
    if (argc > 2) iters       = std::stoul(argv[2]);
    if (argc > 3) n_threads   = std::stoi(argv[3]);

#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#endif

    // ── Génération des particules (rang 0 génère tout, broadcast) ─
    std::vector<particle> all_particles;

#ifdef USE_MPI
    // Le rang 0 génère les particules et les diffuse pour que
    // tous aient le même état initial (reproductibilité).
    if (mpi_rank == 0) {
        all_particles.resize(n_particles);
    }
    // Sérialiser + broadcast
    std::vector<float> global_buf;
    if (mpi_rank == 0) particles_to_buf(all_particles, global_buf);
    else global_buf.resize(n_particles * particle::MPI_FLOATS);
    MPI_Bcast(global_buf.data(),
              static_cast<int>(n_particles * particle::MPI_FLOATS),
              MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) buf_to_particles(global_buf, n_particles, all_particles);

    // Distribuer les particules selon la bande y du rang courant
    float ymin, ymax;
    band_limits(mpi_rank, mpi_size, ymin, ymax);
    std::vector<particle> local_particles;
    for (auto& p : all_particles)
        if (p.y() >= ymin && (p.y() < ymax || mpi_rank == mpi_size-1))
            local_particles.push_back(p);

#else
    all_particles.resize(n_particles);
    std::vector<particle>& local_particles = all_particles;
#endif

    // ── Info ────────────────────────────────────────────────
    if (mpi_rank == 0) {
        std::cout << "=== Barnes-Hut Parallel ===" << std::endl;
        std::cout << "Particules : " << n_particles
                  << "  Itérations : " << iters
                  << "  dt : " << dt
                  << "  theta : " << theta << std::endl;
#ifdef _OPENMP
        std::cout << "OpenMP threads : " << omp_get_max_threads() << std::endl;
#else
        std::cout << "OpenMP : désactivé" << std::endl;
#endif
#ifdef USE_MPI
        std::cout << "MPI rangs : " << mpi_size << std::endl;
#else
        std::cout << "MPI : désactivé" << std::endl;
#endif
    }

    // ── Boucle principale ────────────────────────────────────
    long total_tree_us = 0, total_force_us = 0, total_integrate_us = 0;

    for (std::size_t iter = 0; iter < iters; ++iter) {

        // ── 1. Rassembler toutes les particules pour le qtree global ──
        std::vector<particle> global_view;
#ifdef USE_MPI
        allgather_particles(local_particles, global_view, mpi_size);
#else
        global_view = local_particles; // référence directe en séquentiel
#endif

        // ── 2. Construction du qtree global (séquentielle) ────────────
        struct timeval t0, t1;
        gettimeofday(&t0, nullptr);
        qtree tree(global_view);
        gettimeofday(&t1, nullptr);
        total_tree_us += (t1.tv_sec-t0.tv_sec)*1000000L + (t1.tv_usec-t0.tv_usec);

        // ── 3. Calcul de forces — OPENMP parallel for ─────────────────
        //
        //  Justification de schedule(dynamic) :
        //  Le nombre d'opérations de calc_interaction par particule
        //  dépend de la position relative dans l'arbre (variable).
        //  dynamic évite que les threads terminent à des instants très
        //  différents (déséquilibre). chunk_size=32 : compromis entre
        //  overhead de scheduling et granularité.
        //
        gettimeofday(&t0, nullptr);
        const std::size_t n_local = local_particles.size();

        // Réinitialiser l'accélération avant le calcul
        // (nécessaire car apply_force est cumulatif)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t i = 0; i < n_local; ++i)
            local_particles[i].reset_acceleration();

        // Calcul de force parallèle : chaque particule est traitée
        // indépendamment (pas de dépendance entre les particules locales
        // pour la lecture de l'arbre global en lecture seule).
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 32)
        #endif
        for (std::size_t i = 0; i < n_local; ++i)
            calc_interaction(local_particles[i], &tree, theta);

        gettimeofday(&t1, nullptr);
        total_force_us += (t1.tv_sec-t0.tv_sec)*1000000L + (t1.tv_usec-t0.tv_usec);

        // ── 4. Intégration — OPENMP parallel for ──────────────────────
        //
        //  Justification de schedule(static) :
        //  Coût uniforme par particule → static optimal (zéro overhead
        //  de scheduling dynamique).
        //
        gettimeofday(&t0, nullptr);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t i = 0; i < n_local; ++i)
            local_particles[i].accelerate(dt).move(dt);

        gettimeofday(&t1, nullptr);
        total_integrate_us += (t1.tv_sec-t0.tv_sec)*1000000L + (t1.tv_usec-t0.tv_usec);

        // ── 5. Redistribution MPI après déplacement ───────────────────
#ifdef USE_MPI
        redistribute_particles(local_particles, mpi_rank, mpi_size);
        MPI_Barrier(MPI_COMM_WORLD);  // synchronisation avant la prochaine itération
#endif
    }

    // ── Résultats ────────────────────────────────────────────
    if (mpi_rank == 0) {
        std::cout << "\n-- Timings cumulés (" << iters << " itérations) --" << std::endl;
        std::cout << "  Construction arbre : " << total_tree_us      << " us" << std::endl;
        std::cout << "  Calcul de forces   : " << total_force_us     << " us" << std::endl;
        std::cout << "  Intégration        : " << total_integrate_us << " us" << std::endl;
        std::cout << "  TOTAL              : "
                  << (total_tree_us+total_force_us+total_integrate_us) << " us" << std::endl;
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
}
