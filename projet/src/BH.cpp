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

namespace CONSTANTS {
    constexpr float G = 6.6742e-11f;
}

// Forward declarations
class particle;

// Calculate the cumulative error (squared) between two containers
template <class container1, class container2>
float get_cumulative_error(const container1& v1, const container2& v2) {
    float err = 0.f;
    typename container1::const_iterator it1;
    typename container2::const_iterator it2;
    for (it1 = v1.cbegin(), it2 = v2.cbegin(); it1 != v1.cend() && it2 != v2.cend(); ++it1, ++it2) {
        err += std::powf(it1->x() - it2->x(), 2.f) + std::powf(it1->y() - it2->y(), 2.f);
    }
    return err;
}

// Print the content of two containers
template <class container1, class container2>
std::ostream& print(std::ostream& os, const container1& v1, const container2& v2) {
    typename container1::const_iterator it1;
    typename container2::const_iterator it2;
    for (it1 = v1.cbegin(), it2 = v2.cbegin(); it1 != v1.cend() && it2 != v2.cend(); ++it1, ++it2) {
        os << *it1 << std::endl << *it2 << std::endl;
    }
    return os;
}

struct Point2D {
    private:
        // Print the coordinates with a fixed width
        struct print_coord {
            float coord;
            friend std::ostream& operator<<(std::ostream& os, const print_coord& c) {
                const auto default_precision{std::cout.precision()};
                return os << std::right << std::setw(8) << std::setprecision(5)
                    << c.coord << std::setw(0) << std::setprecision(default_precision);
            }
        };
    public:
    float x = 0.f, y = 0.f;
    friend std::ostream& operator<<(std::ostream& os, const Point2D& pt) {
        return os << '(' << print_coord{pt.x} << ',' << print_coord{pt.y} << ')';
    }
    Point2D& operator+=(const Point2D& vec) {
        x += vec.x;
        y += vec.y;
        return *this;
    }
    friend Point2D operator+(const Point2D& vec1, const Point2D& vec2) {
        return {vec1.x + vec2.x, vec1.y + vec2.y};
    }
    friend Point2D&& operator+(const Point2D& vec1, Point2D&& vec2) {
        vec2 += vec1;
        return std::move(vec2);
    }
    // Commutative operator
    friend Point2D&& operator+(Point2D&& vec1, const Point2D& vec2) {
        return vec2 + std::move(vec1);
    }
    friend Point2D&& operator+(Point2D&& vec1, Point2D&& vec2) {
        return vec1 + std::move(vec2);
    }
    Point2D operator-(const Point2D& p) const {
        return {x - p.x, y - p.y};
    }
    bool operator<(const Point2D& p) const {
        return (x < p.x || y < p.y);
    }
};

struct Vector2D: Point2D {
    // Force de p2 sur p1
    Vector2D() = default;
    Vector2D(float x, float y): Point2D({x,y}) {}
    Vector2D(const Point2D& p1, const Point2D& p2): Point2D(p1-p2) {}
    Vector2D(const Vector2D&) = default;
    Vector2D(Vector2D&&) = default;
    Vector2D& operator=(const Vector2D&) = default;
    Vector2D& operator=(Vector2D&&) = default;
    float magnitude() const {
        return std::hypotf(x, y);
    }
    Vector2D unit() const {
        return Vector2D(x/magnitude(), y/magnitude());
    }
    Vector2D operator*(float accel) const {
        return Vector2D(x * accel, y * accel);
    }
    // Commutative version of the operator
    friend Vector2D operator*(float accel, const Vector2D& vec) {
        return vec * accel;
    }
};

class particle {
    Point2D pos_;
    Vector2D speed_;
    float mass_ = 0.f;
    Vector2D acceleration_;
    private:
        // Print the mass with a fixed width
        struct print_mass {
            float mass;
            friend std::ostream& operator<<(std::ostream& os, const print_mass& c) {
                const auto default_precision{std::cout.precision()};
                return os << std::left << std::setw(2) << std::setprecision(5)
                    << c.mass << std::setw(0) << std::setprecision(default_precision);
            }
        };
    public:
    // Initialize a particle with random coordinates (uniform), mass (normal) and speed (normal)
    particle(): pos_({MyRNG::get_coord(), MyRNG::get_coord()}), speed_({MyRNG::get_speed(), MyRNG::get_speed()}), mass_(MyRNG::get_mass()) {}
    particle(const particle&) = default;
    particle(particle&&) = default;
    particle& operator=(const particle&) = default;
    particle& operator=(particle&&) = default;
    particle(const Point2D& p): pos_(p) {}
    const float& x() const { return pos_.x; }
    const float& y() const { return pos_.y; }
    const float& mass() const { return mass_; }
    bool has_mass() const { return 0.f < mass_; }
    particle& operator+=(const particle& p) {
        const float new_mass = mass_ + p.mass_;
        pos_.x = (pos_.x * mass_ + p.pos_.x * p.mass_) / new_mass;
        pos_.y = (pos_.y * mass_ + p.pos_.y * p.mass_) / new_mass;
        mass_  = new_mass;
        return *this;
    }
    particle& apply_force(const particle& p) {
        Vector2D res(pos_, p.pos_);
        const float mag = res.magnitude();
        const float acceleration = -1 * CONSTANTS::G * p.mass_ / std::powf(mag, 2.f);
        Vector2D unit_vector = res.unit();
        acceleration_ += unit_vector * acceleration;
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
    private:
    static inline long long get_bounces(float val, float limit) {
        return 1.f + val / (limit * 2.f);
    }
    static inline float maybe_invert(float val, long long n_bounces) {
        return n_bounces & 1 ? -1.f * val : val;
    }
    static inline float get_rebound_pos(float val, float limit, long long bounces) {
        return limit - std::fmod(val + 2.f * ((bounces >> 1) << 1), limit);
    }
    static inline void update_with_rebounds(float& pos, float& speed, float limit) {
        const long long bounces = get_bounces(pos, limit);
        speed = maybe_invert(speed, bounces);
        pos = get_rebound_pos(pos, limit, bounces);
    }
    static inline void check_and_update_with_rebounds(float& pos, float& speed, float limit) {
        if (pos < -limit) update_with_rebounds(pos, speed, -limit);
        if (pos >  limit) update_with_rebounds(pos, speed,  limit);
    }
    private:
    class MyRNG {
        static std::mt19937 generator;
        static float get_normal_float() {
            static std::normal_distribution<float> distribution(0.0f, 1.0f);
            return distribution(generator);
        }
        static float get_uni_float() {
            std::uniform_real_distribution<float> distribution(-.5f, std::nextafter(.5f, FLT_MAX));
            return distribution(generator);
        }
        public:
        static float get_mass() { return std::abs(get_normal_float()); }
        static float get_coord() { return get_uni_float(); }
        static float get_speed() { return get_normal_float() * .1f; }
    };
    public:
    friend std::ostream& operator<<(std::ostream& os, const particle& p) {
        return os << "{ " << p.pos_ << ": " << print_mass{p.mass_} << " }";
    }
    bool operator<(const particle& p) const {
        return pos_ < p.pos_;
    }
};
std::mt19937 particle::MyRNG::generator(std::random_device{}());

class qtree {
    qtree *ne_ = nullptr, *nw_ = nullptr, *se_ = nullptr, *sw_ = nullptr;
    Point2D pos_ = {0.f, 0.f};
    // Union pour assurer que les particules ne sont pas modifiables, seules les "méta-particules" le sont.
    union {
        const particle *leaf_res_;
        particle *res_;
    };
    float width_ = 1.f, height_ = 1.f;
    qtree(const particle *part, const Point2D& p, float w, float h) : pos_(p), leaf_res_(part),
        width_(0.f != w ? w : throw std::domain_error("null width")),
        height_(0.f != h ? h : throw std::domain_error("null height")) {}
    public:
    // Construit l'arbre à partir d'un container (quelqu'il soit)
    template <class T> qtree(const T& container) : res_(new particle(pos_)) {
        for (typename T::const_iterator it = container.cbegin(); it != container.cend(); ++it) {
            add_particle(&(*it));
        }
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
            res_ = new particle(*leaf_res_); // copy resulting particle
        }
        *res_ += *p;
        insert_particle(p);
    }
    bool has_particle() const { return res_->has_mass(); }
    bool is_leaf() const { return !(ne_ || nw_ || se_ || sw_); }
    const particle& leaf_res() const { return *leaf_res_; }
    const particle& res() const { return *res_; }
    friend std::ostream& operator<<(std::ostream& os, const qtree& tree) {
        return qtree::print(os, tree, 0);
    }
    private:
    // Affichage
    struct indent {
        std::size_t lvl_;
        indent(std::size_t lvl): lvl_(lvl) {}
        friend std::ostream& operator<<(std::ostream& os, const indent& indent) { for (std::size_t i = 0; i < indent.lvl_; ++i) os << "|\t";  return os; };
    };
    static std::ostream& print(std::ostream& os, const qtree* tree, std::size_t lvl) { return (nullptr == tree) ? (os << "nil") : print(os, *tree, lvl); }
    static std::ostream& print(std::ostream& os, const qtree& tree, std::size_t lvl) {
        os << "{ " << *tree.res_ << ' ';
        if (!tree.is_leaf()) {
            os << std::endl << indent(lvl+1) << "ne:";
            print(os, tree.ne_, lvl+1) << std::endl << indent(lvl+1) << "nw:";
            print(os, tree.nw_, lvl+1) << std::endl << indent(lvl+1) << "se:";
            print(os, tree.se_, lvl+1) << std::endl << indent(lvl+1) << "sw:";
            print(os, tree.sw_, lvl+1) << std::endl << indent(lvl);
        }
        os << '}';
        return os;
    }
    // Insertion dans l'arbre
    void insert_particle(const particle *p) {
        const float new_width   = width_  * .5f;
        const float new_height  = height_ * .5f;
        const Point2D center = { new_width * .5f, new_height * .5f };
        // South-West
        if      (p->x() <  pos_.x && p->y() <  pos_.y) {
            if (!sw_) sw_ = new qtree(p, pos_ + Vector2D{-center.x,-center.y}, new_width, new_height);
            else sw_->add_particle(p);
        }
        // North-West
        else if (p->x() <  pos_.x && p->y() >= pos_.y) {
            if (!nw_) nw_ = new qtree(p, pos_ + Vector2D{-center.x, center.y}, new_width, new_height);
            else nw_->add_particle(p);
        }
        // South-East
        else if (p->x() >= pos_.x && p->y() <  pos_.y) {
            if (!se_) se_ = new qtree(p, pos_ + Vector2D{ center.x,-center.y}, new_width, new_height);
            else se_->add_particle(p);
        }
        // North-East
        else if (p->x() >= pos_.x && p->y() >= pos_.y) {
            if (!ne_) ne_ = new qtree(p, pos_ + Vector2D{ center.x, center.y}, new_width, new_height);
            else ne_->add_particle(p);
        }
        else ; // silence warning
    }
    public:
    friend void calc_interaction(particle& p, const qtree* tree, const float theta) {
        if (tree) {
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
    }
};

class timer {
        struct timeval start, end;
        long elapsed_time(const struct timeval& start, const struct timeval& end) {
            const long seconds = end.tv_sec - start.tv_sec;
            const long microseconds = end.tv_usec - start.tv_usec;
            const long elapsed = seconds * 1000000 + microseconds;
            return elapsed;
        }
    public:
        timer() { gettimeofday(&start, NULL); }
        ~timer() {
            gettimeofday(&end, NULL);
            const long elapsed = elapsed_time(start, end);
            std::cout << "Elapsed time: " << elapsed << " microseconds" << std::endl;
        }
};

int main(int argc, const char** argv) {
    // Generate a random seed at the start of the program using random_device
    std::random_device rd;
    unsigned long long global_seed = rd();  // This will be the global seed
    (void) global_seed;

    std::size_t n_particles = 100000;
    if (argc > 1) n_particles = std::stoul(argv[1]);
    std::size_t iters = 10;
    // particle *tab = new particle[n_particles];
    std::vector<particle> tab(n_particles);
    // std::sort(tab.begin(), tab.end());
    qtree tree = qtree(tab);
    for (const auto& p: tab) std::cerr << p << std::endl;
    std::cerr << std::endl;
    std::cerr << tree << std::endl;

    std::vector<particle> copy_tab(tab);

    const float dt = .5f; // half a second
    std::cout << "nb iterations: " << iters << ", nb particules: " << tab.size() << ", dt: " << dt << std::endl;
    {

        timer t;
        for (std::size_t i = 0; i < iters; ++i) {
            qtree tree = qtree(tab);
            for (auto& p: tab) calc_interaction(p, &tree, .5f);
            for (auto& p: tab) p.accelerate(dt).move(dt);
        }
        std::cout << "         BH: ";
    }

    {
        timer t;
        for (std::size_t i = 0; i < iters; ++i) {
            for (std::size_t i = 0; i < copy_tab.size(); ++i) {
                for (std::size_t j = i+1; j < copy_tab.size(); ++j) {
                    copy_tab[i].apply_force(copy_tab[j]).accelerate(dt);
                    copy_tab[j].apply_force(copy_tab[i]).accelerate(dt);
                }
            }
            for (auto& p: copy_tab) p.move(dt);
        }
        std::cout << "brute force: ";
    }

    std::cout << "cumulative squared error: " << get_cumulative_error(tab, copy_tab) << std::endl;

    print(std::cerr, tab, copy_tab);

    return EXIT_SUCCESS;
}
