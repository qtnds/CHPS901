// ============================================================
//  common.hpp — Types partagés par toutes les versions BH
//  (Point2D, Vector2D, particle, qtree, timer, MPI helpers)
//  Les #pragma omp sont placés dans chaque version, pas ici.
// ============================================================
#pragma once
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <cfloat>
#include <stdexcept>
#include <sys/time.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

namespace CONSTANTS { constexpr float G = 6.6742e-11f; }

// ── Point2D ──────────────────────────────────────────────────
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

// ── Vector2D ─────────────────────────────────────────────────
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

// ── particle ─────────────────────────────────────────────────
class particle {
    Point2D  pos_;
    Vector2D speed_;
    float    mass_ = 0.f;
    Vector2D acceleration_;

    struct print_mass {
        float mass;
        friend std::ostream& operator<<(std::ostream& os, const print_mass& c) {
            const auto dp{std::cout.precision()};
            return os << std::left << std::setw(2) << std::setprecision(5)
                      << c.mass << std::setw(0) << std::setprecision(dp);
        }
    };

    class MyRNG {
        static std::mt19937 generator;
        static float get_normal_float() {
            static std::normal_distribution<float> d(0.f,1.f);
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
    particle()
        : pos_({MyRNG::get_coord(), MyRNG::get_coord()})
        , speed_({MyRNG::get_speed(), MyRNG::get_speed()})
        , mass_(MyRNG::get_mass()) {}
    particle(const particle&) = default;
    particle(particle&&) = default;
    particle& operator=(const particle&) = default;
    particle& operator=(particle&&) = default;
    explicit particle(const Point2D& p) : pos_(p) {}

    const float& x()    const { return pos_.x; }
    const float& y()    const { return pos_.y; }
    const float& mass() const { return mass_; }
    bool has_mass()     const { return 0.f < mass_; }

    // MPI flat layout : [px, py, sx, sy, m, ax, ay]
    static constexpr int MPI_FLOATS = 7;
    void pack(float* b) const {
        b[0]=pos_.x; b[1]=pos_.y; b[2]=speed_.x; b[3]=speed_.y;
        b[4]=mass_;  b[5]=acceleration_.x; b[6]=acceleration_.y;
    }
    void unpack(const float* b) {
        pos_.x=b[0]; pos_.y=b[1]; speed_.x=b[2]; speed_.y=b[3];
        mass_=b[4]; acceleration_.x=b[5]; acceleration_.y=b[6];
    }

    particle& operator+=(const particle& p) {
        const float nm = mass_ + p.mass_;
        pos_.x = (pos_.x*mass_ + p.pos_.x*p.mass_) / nm;
        pos_.y = (pos_.y*mass_ + p.pos_.y*p.mass_) / nm;
        mass_  = nm;
        return *this;
    }
    particle& apply_force(const particle& p) {
        Vector2D res(pos_, p.pos_);
        const float mag = res.magnitude();
        const float acc = -1.f * CONSTANTS::G * p.mass_ / (mag*mag);
        acceleration_ += res.unit() * acc;
        return *this;
    }
    particle& accelerate(float dt) { speed_ += acceleration_ * dt; return *this; }
    particle& move(float dt) {
        pos_ += speed_ * dt;
        do_rebounds(pos_.x, speed_.x, .5f);
        do_rebounds(pos_.y, speed_.y, .5f);
        return *this;
    }
    void reset_acceleration() { acceleration_ = Vector2D(0.f,0.f); }
    bool operator<(const particle& p) const { return pos_ < p.pos_; }
    friend std::ostream& operator<<(std::ostream& os, const particle& p) {
        return os << "{ " << p.pos_ << ": " << print_mass{p.mass_} << " }";
    }
private:
    static void do_rebounds(float& pos, float& spd, float lim) {
        auto rebound = [](float& pos, float& spd, float lim){
            long long b = (long long)(1.f + pos/(lim*2.f));
            if (b & 1) spd = -spd;
            pos = lim - std::fmod(pos + 2.f*(float)((b>>1)<<1), lim);
        };
        if (pos < -lim) rebound(pos, spd, -lim);
        if (pos >  lim) rebound(pos, spd,  lim);
    }
};
inline std::mt19937 particle::MyRNG::generator(std::random_device{}());

// ── qtree ─────────────────────────────────────────────────────
class qtree {
    qtree *ne_=nullptr,*nw_=nullptr,*se_=nullptr,*sw_=nullptr;
    Point2D pos_ = {0.f,0.f};
    union { const particle* leaf_res_; particle* res_; };
    float width_=1.f, height_=1.f;

    qtree(const particle* part, const Point2D& p, float w, float h)
        : pos_(p), leaf_res_(part)
        , width_ (0.f!=w ? w : throw std::domain_error("null width"))
        , height_(0.f!=h ? h : throw std::domain_error("null height")) {}
public:
    template <class T>
    explicit qtree(const T& c) : res_(new particle(pos_)) {
        for (auto it=c.cbegin(); it!=c.cend(); ++it) add_particle(&(*it));
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
    bool is_leaf() const { return !(ne_||nw_||se_||sw_); }
    const particle& leaf_res() const { return *leaf_res_; }
    const particle& res()      const { return *res_; }

private:
    void insert_particle(const particle* p) {
        const float nw=width_*.5f, nh=height_*.5f;
        const Point2D c={nw*.5f, nh*.5f};
        if      (p->x()<pos_.x && p->y()<pos_.y)  { if(!sw_) sw_=new qtree(p,pos_+Vector2D{-c.x,-c.y},nw,nh); else sw_->add_particle(p); }
        else if (p->x()<pos_.x && p->y()>=pos_.y) { if(!nw_) nw_=new qtree(p,pos_+Vector2D{-c.x, c.y},nw,nh); else nw_->add_particle(p); }
        else if (p->x()>=pos_.x && p->y()<pos_.y) { if(!se_) se_=new qtree(p,pos_+Vector2D{ c.x,-c.y},nw,nh); else se_->add_particle(p); }
        else                                       { if(!ne_) ne_=new qtree(p,pos_+Vector2D{ c.x, c.y},nw,nh); else ne_->add_particle(p); }
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
            } else p.apply_force(tree->res());
        }
    }
};

// ── timer léger ───────────────────────────────────────────────
struct ticker {
    struct timeval s_;
    ticker() { gettimeofday(&s_, nullptr); }
    long lap() const {
        struct timeval n; gettimeofday(&n, nullptr);
        return (n.tv_sec-s_.tv_sec)*1000000L + (n.tv_usec-s_.tv_usec);
    }
};

// ── Sortie machine (parsée par benchmark.sh / analyse.py) ─────
// Format : TIMING tree_us=X force_us=X integrate_us=X total_us=X
inline void print_timings(long tree, long force, long integrate, int rank=0) {
    if (rank==0)
        std::cout << "TIMING"
                  << " tree_us="      << tree
                  << " force_us="     << force
                  << " integrate_us=" << integrate
                  << " total_us="     << (tree+force+integrate)
                  << std::endl;
}

// ── MPI helpers ───────────────────────────────────────────────
#ifdef USE_MPI
inline void particles_to_buf(const std::vector<particle>& v, std::vector<float>& buf) {
    buf.resize(v.size()*particle::MPI_FLOATS);
    for (std::size_t i=0;i<v.size();++i) v[i].pack(&buf[i*particle::MPI_FLOATS]);
}
inline void buf_to_particles(const std::vector<float>& buf, std::size_t n,
                              std::vector<particle>& v) {
    v.resize(n);
    for (std::size_t i=0;i<n;++i) {
        particle p(Point2D{}); p.unpack(&buf[i*particle::MPI_FLOATS]); v[i]=std::move(p);
    }
}
inline void band_limits(int rank, int size, float& ymin, float& ymax) {
    const float band=1.f/(float)size;
    ymin=-0.5f+rank*band; ymax=ymin+band;
}
inline int owner_rank(float y, int size) {
    int r=(int)((y+0.5f)*size); return std::max(0,std::min(size-1,r));
}
inline void allgather_particles(const std::vector<particle>& local,
                                 std::vector<particle>& global, int mpi_size) {
    int ln=(int)local.size();
    std::vector<int> all_n(mpi_size);
    MPI_Allgather(&ln,1,MPI_INT,all_n.data(),1,MPI_INT,MPI_COMM_WORLD);
    std::vector<float> lbuf; particles_to_buf(local,lbuf);
    std::vector<int> counts(mpi_size), displs(mpi_size,0);
    for (int i=0;i<mpi_size;++i) counts[i]=all_n[i]*particle::MPI_FLOATS;
    for (int i=1;i<mpi_size;++i) displs[i]=displs[i-1]+counts[i-1];
    const int tf=displs[mpi_size-1]+counts[mpi_size-1];
    std::vector<float> gbuf(tf);
    MPI_Allgatherv(lbuf.data(),(int)lbuf.size(),MPI_FLOAT,
                   gbuf.data(),counts.data(),displs.data(),MPI_FLOAT,MPI_COMM_WORLD);
    buf_to_particles(gbuf, tf/particle::MPI_FLOATS, global);
}
inline void redistribute_particles(std::vector<particle>& local, int /*rank*/, int mpi_size) {
    std::vector<std::vector<particle>> ts(mpi_size);
    for (auto& p:local) ts[owner_rank(p.y(),mpi_size)].push_back(p);
    std::vector<float> sbuf; std::vector<int> sc(mpi_size,0);
    for (int r=0;r<mpi_size;++r) {
        sc[r]=(int)ts[r].size()*particle::MPI_FLOATS;
        std::vector<float> tmp; particles_to_buf(ts[r],tmp);
        sbuf.insert(sbuf.end(),tmp.begin(),tmp.end());
    }
    std::vector<int> rc(mpi_size);
    MPI_Alltoall(sc.data(),1,MPI_INT,rc.data(),1,MPI_INT,MPI_COMM_WORLD);
    std::vector<int> sd(mpi_size,0),rd(mpi_size,0);
    for (int i=1;i<mpi_size;++i) { sd[i]=sd[i-1]+sc[i-1]; rd[i]=rd[i-1]+rc[i-1]; }
    const int tr=rd[mpi_size-1]+rc[mpi_size-1];
    std::vector<float> rbuf(tr);
    MPI_Alltoallv(sbuf.data(),sc.data(),sd.data(),MPI_FLOAT,
                  rbuf.data(),rc.data(),rd.data(),MPI_FLOAT,MPI_COMM_WORLD);
    buf_to_particles(rbuf, tr/particle::MPI_FLOATS, local);
}
#endif // USE_MPI
