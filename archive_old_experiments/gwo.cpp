#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Hook in your original benchmarking code, but rename its main to baseline_main
// -----------------------------------------------------------------------------
#define main baseline_main
#include "benchmarking_code.cpp"
#undef main

// -----------------------------------------------------------------------------
// Simple heap usage tracker (same pattern as abc.cpp / pso.cpp)
// -----------------------------------------------------------------------------
namespace memtrack {
    struct Header {
        size_t size;
        void*  raw;
    };

    static atomic<size_t> g_live{0}, g_total{0}, g_peak{0};

    static inline void note_alloc(size_t sz){
        g_live.fetch_add(sz, memory_order_relaxed);
        g_total.fetch_add(sz, memory_order_relaxed);
        size_t live = g_live.load(memory_order_relaxed);
        size_t prev = g_peak.load(memory_order_relaxed);
        while (live > prev &&
               !g_peak.compare_exchange_weak(prev, live, memory_order_relaxed)) {}
    }
    static inline void note_free(size_t sz){
        g_live.fetch_sub(sz, memory_order_relaxed);
    }

    static inline void* alloc_block(size_t sz, size_t alignment){
        size_t need = sz + alignment + sizeof(Header);
        char* raw = (char*)std::malloc(need);
        if (!raw) throw std::bad_alloc();
        char* start = raw + sizeof(Header);
        uintptr_t addr = reinterpret_cast<uintptr_t>(start);
        size_t pad = (alignment==0?0: (alignment - (addr % alignment)) % alignment);
        char* aligned = start + pad;
        Header* h = reinterpret_cast<Header*>(aligned) - 1;
        h->size = sz; h->raw = raw;
        note_alloc(sz);
        return aligned;
    }
    static inline void free_block(void* p){
        if (!p) return;
        Header* h = reinterpret_cast<Header*>(p) - 1;
        note_free(h->size);
        std::free(h->raw);
    }

    struct Scope {
        size_t base_live=0, base_total=0, base_peak=0;
        Scope(){ reset(); }
        void reset(){
            base_live  = g_live.load();
            base_total = g_total.load();
            base_peak  = g_peak.load();
        }
        size_t live_delta() const {
            size_t cur = g_live.load();
            return (cur>=base_live)?  cur-base_live : 0;
        }
        size_t total_delta() const {
            size_t cur = g_total.load();
            return (cur>=base_total)? cur-base_total : 0;
        }
        size_t peak_delta() const {
            size_t cur = g_peak.load();
            return (cur>base_peak)? cur-base_peak : 0;
        }
    };
} // namespace memtrack

// ---- global new/delete (regular, array, sized, aligned) ----
void* operator new(size_t sz){
    return ::memtrack::alloc_block(sz, alignof(max_align_t));
}
void operator delete(void* p) noexcept{
    ::memtrack::free_block(p);
}
void* operator new[](size_t sz){
    return ::memtrack::alloc_block(sz, alignof(max_align_t));
}
void operator delete[](void* p) noexcept{
    ::memtrack::free_block(p);
}
void operator delete(void* p, size_t) noexcept { ::memtrack::free_block(p); }
void operator delete[](void* p, size_t) noexcept { ::memtrack::free_block(p); }

#if (__cpp_aligned_new >= 201606)
void* operator new(size_t sz, std::align_val_t al){
    return ::memtrack::alloc_block(sz, static_cast<size_t>(al));
}
void operator delete(void* p, std::align_val_t) noexcept{
    ::memtrack::free_block(p);
}
void* operator new[](size_t sz, std::align_val_t al){
    return ::memtrack::alloc_block(sz, static_cast<size_t>(al));
}
void operator delete[](void* p, std::align_val_t) noexcept{
    ::memtrack::free_block(p);
}
void operator delete(void* p, size_t, std::align_val_t) noexcept{
    ::memtrack::free_block(p);
}
void operator delete[](void* p, size_t, std::align_val_t) noexcept{
    ::memtrack::free_block(p);
}
#endif

// -----------------------------------------------------------------------------
// TIMING + ANTI-OPT HELPERS (same idea as in pso.cpp / abc.cpp)
// -----------------------------------------------------------------------------
using clk = chrono::steady_clock;

static inline long long span_ns(const clk::time_point& a, const clk::time_point& b){
    return chrono::duration_cast<chrono::nanoseconds>(b - a).count();
}

// Accept const volatile so we can pass &volatile_var without cast.
static inline void do_not_optimize_away(const volatile void* p){
#if defined(__GNUG__) || defined(__clang__)
    asm volatile("" : : "g"(p) : "memory");
#else
    atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

template<class Fn>
static long long avg_ns_runs(Fn body, int R){
    body(); // warmup
    auto t0 = clk::now();
    for (int r=0; r<R; ++r) body();
    auto t1 = clk::now();
    return span_ns(t0, t1) / max(1, R);
}

template<class Fn>
static double avg_loop_ns(Fn loop, int R){
    loop(); // warmup
    auto t0 = clk::now();
    for (int r=0; r<R; ++r) loop();
    auto t1 = clk::now();
    return (double)span_ns(t0, t1) / (double)max(1, R);
}

// -----------------------------------------------------------------------------
// approx_bytes SFINAE (non-const-friendly, same fix as pso.cpp)
// -----------------------------------------------------------------------------
template<class T>
auto try_approx_bytes(T& t, int) -> decltype(t.approx_bytes(), size_t{}) {
    return t.approx_bytes();
}
template<class T>
size_t try_approx_bytes(T&, long){
    return 0;
}

// -----------------------------------------------------------------------------
// BENCH TYPES
// -----------------------------------------------------------------------------
struct BenchConfig {
    int K;
    int THR;
    int Csize;
};

struct BenchResult {
    // time (ms or ms/query)
    double fqa_build_ms=0.0, bt_build_ms=0.0, qf_build_ms=0.0;
    double fqa_q_ms=0.0,     bt_q_ms=0.0,    qf_q_ms=0.0;

    // resident sizes (bytes)
    size_t fqa_mem=0,        bt_mem=0,       qf_mem=0;

    // ns (detail)
    double fqa_build_ns=0.0, bt_build_ns=0.0, qf_build_ns=0.0;
    double fqa_q_ns=0.0,     bt_q_ns=0.0,     qf_q_ns=0.0;

    // memory peaks per phase
    size_t fqa_build_peak=0, bt_build_peak=0, qf_build_peak=0;
    size_t fqa_query_peak=0, bt_query_peak=0, qf_query_peak=0;
    size_t fqa_query_total=0, bt_query_total=0, qf_query_total=0;
    size_t fqa_query_live_delta=0, bt_query_live_delta=0, qf_query_live_delta=0;
};

// -----------------------------------------------------------------------------
// QFast helpers (build + find) – same pattern as pso.cpp
// -----------------------------------------------------------------------------
struct QFBuild {
    // prefer build(students, Csize)
    template<class T>
    static auto build(T& t, const vector<Student>& s, int C, int)
        -> decltype(t.build(s, C), void())
    {
        t.build(s, C);
    }
    // fallback to build(students)
    template<class T>
    static auto build(T& t, const vector<Student>& s, int, long)
        -> decltype(t.build(s), void())
    {
        t.build(s);
    }
};

// countish(): robust result-counter
// 1) If T is integral -> return value
template<class T, class = typename enable_if<
    is_integral<typename decay<T>::type>::value>::type>
static inline size_t countish(T v){ return (size_t)v; }

// 2) If T has .size() -> return size()
template<class T>
static inline auto countish(const T& v) -> decltype(v.size(), size_t{}){
    return (size_t)v.size();
}

// 3) Fallback -> treat as 1 result
static inline size_t countish(...){ return 1; }

// Prefer find_reg, else find_exact, else find
template<class QF>
auto qf_try(QF& qf, const string& s, int) -> decltype(qf.find_reg(s)) {
    return qf.find_reg(s);
}
template<class QF>
auto qf_try(QF& qf, const string& s, long) -> decltype(qf.find_exact(s)) {
    return qf.find_exact(s);
}
template<class QF>
auto qf_try(QF& qf, const string& s, char) -> decltype(qf.find(s)) {
    return qf.find(s);
}

template<class QF>
static size_t qf_query_count(QF& qf, const string& s){
    auto r = qf_try(qf, s, 0);
    return countish(r);
}

// -----------------------------------------------------------------------------
// EVALUATE ONE CONFIG (time + memory) – lifted from your PSO harness
// -----------------------------------------------------------------------------
static BenchResult evaluate_config(const vector<Student>& students,
                                   const BenchConfig& cfg,
                                   int Q, uint32_t seed, double EPS)
{
    BenchResult R{};
    static volatile size_t sink_guard = 0;

    // ---------- FQA ----------
    {
        memtrack::Scope msb;
        FQAIndex fqa;
        fqa.build(students, cfg.K);
        R.fqa_mem = try_approx_bytes(fqa, 0);
        R.fqa_build_peak = msb.peak_delta();

        int build_reps = 20;
        auto build_ns = avg_ns_runs([&](){
            FQAIndex tmp;
            tmp.build(students, cfg.K);
            sink_guard ^= try_approx_bytes(tmp, 0) + 1;
            do_not_optimize_away(&tmp);
        }, build_reps);
        R.fqa_build_ns = (double)build_ns;
        R.fqa_build_ms = (double)build_ns / 1e6;

        mt19937 rng(seed);
        uniform_int_distribution<int> di(0, (int)students.size()-1);
        vector<double> cg_queries(Q);
        for (int i=0;i<Q;i++) cg_queries[i] = students[ di(rng) ].cgpa;

        auto one_loop = [&](){
            for (int i=0;i<Q;i++){
                volatile auto res = fqa.query_exact(cg_queries[i], EPS);
                (void)res;
            }
        };
        memtrack::Scope msq;
        one_loop();
        R.fqa_query_peak       = msq.peak_delta();
        R.fqa_query_total      = msq.total_delta();
        R.fqa_query_live_delta = msq.live_delta();

        int loop_reps = 60;
        double ns_full = avg_loop_ns(one_loop, loop_reps);
        double ns_per_query = ns_full / max(1, Q);
        R.fqa_q_ns = ns_per_query;
        R.fqa_q_ms = ns_per_query / 1e6;
        sink_guard ^= (size_t)R.fqa_q_ns + 7;
    }

    // ---------- BurstTrie ----------
    {
        memtrack::Scope msb;
        BurstTrie bt;
        bt.build(students, cfg.THR);
        R.bt_mem = try_approx_bytes(bt, 0);
        R.bt_build_peak = msb.peak_delta();

        int build_reps = 20;
        auto build_ns = avg_ns_runs([&](){
            BurstTrie tmp; tmp.build(students, cfg.THR);
            sink_guard ^= try_approx_bytes(tmp, 0) + 3;
            do_not_optimize_away(&tmp);
        }, build_reps);
        R.bt_build_ns = (double)build_ns;
        R.bt_build_ms = (double)build_ns / 1e6;

        mt19937 rng(seed+1);
        uniform_int_distribution<int> di(0, (int)students.size()-1);
        vector<string> name_queries(Q);
        for (int i=0;i<Q;i++) name_queries[i] = students[ di(rng) ].name;

        auto one_loop = [&](){
            for (int i=0;i<Q;i++){
                volatile auto res = bt.find_exact(name_queries[i]);
                (void)res;
            }
        };
        memtrack::Scope msq;
        one_loop();
        R.bt_query_peak       = msq.peak_delta();
        R.bt_query_total      = msq.total_delta();
        R.bt_query_live_delta = msq.live_delta();

        int loop_reps = 60;
        double ns_full = avg_loop_ns(one_loop, loop_reps);
        double ns_per_query = ns_full / max(1, Q);
        R.bt_q_ns = ns_per_query;
        R.bt_q_ms = ns_per_query / 1e6;
        sink_guard ^= (size_t)R.bt_q_ns + 11;
    }

    // ---------- QFastTrie ----------
    {
        memtrack::Scope msb;
        QFastTrie qf;
        QFBuild::build(qf, students, cfg.Csize, 0); // prefer (s, C), else (s)
        R.qf_mem = try_approx_bytes(qf, 0);
        R.qf_build_peak = msb.peak_delta();

        int build_reps = 20;
        auto build_ns = avg_ns_runs([&](){
            QFastTrie tmp;
            QFBuild::build(tmp, students, cfg.Csize, 0);
            do_not_optimize_away(&tmp);
        }, build_reps);
        R.qf_build_ns = (double)build_ns;
        R.qf_build_ms = (double)build_ns / 1e6;

        mt19937 rng(seed+2);
        uniform_int_distribution<int> di(0, (int)students.size()-1);
        vector<string> reg_queries(Q);
        for (int i=0;i<Q;i++) reg_queries[i] = students[ di(rng) ].reg_no;

        auto one_loop = [&](){
            for (int i=0;i<Q;i++){
                volatile size_t cnt = qf_query_count(qf, reg_queries[i]);
                (void)cnt;
            }
        };
        memtrack::Scope msq;
        one_loop();
        R.qf_query_peak       = msq.peak_delta();
        R.qf_query_total      = msq.total_delta();
        R.qf_query_live_delta = msq.live_delta();

        int loop_reps = 60;
        double ns_full = avg_loop_ns(one_loop, loop_reps);
        double ns_per_query = ns_full / max(1, Q);
        R.qf_q_ns = ns_per_query;
        R.qf_q_ms = ns_per_query / 1e6;
        sink_guard ^= (size_t)R.qf_q_ns + 17;
    }

    (void)sink_guard;
    return R;
}

// -----------------------------------------------------------------------------
// Cost model: combine build time, query time, and memory
// You can tweak weights via CLI if you like.
// -----------------------------------------------------------------------------
struct CostModel {
    double w_build = 1.0;    // weight for total build time (ms)
    double w_query = 5.0;    // weight for total query time (ms)
    double w_mem   = 0.001;  // weight for memory (MB)

    double operator()(const BenchResult& r) const {
        double build_sum_ms =
            r.fqa_build_ms + r.bt_build_ms + r.qf_build_ms;

        double query_sum_ms =
            r.fqa_q_ms + r.bt_q_ms + r.qf_q_ms;

        double mem_mb =
            (double)(r.fqa_mem + r.bt_mem + r.qf_mem) / (1024.0 * 1024.0);

        return w_build * build_sum_ms +
               w_query * query_sum_ms +
               w_mem   * mem_mb;
    }
};
// -----------------------------------------------------------------------------
// CLI helpers for this GWO wrapper
// -----------------------------------------------------------------------------
static void print_usage(){
    cout << "Usage:\n";
    cout << "  ./gwo [baseline args]               # run original benchmark main\n";
    cout << "  ./gwo --gwo=1 [opts]                # run Grey Wolf Optimiser\n\n";
    cout << "Baseline args (forwarded to baseline_main):\n";
    cout << "  --k=<int>       FQA pivots (K)\n";
    cout << "  --thr=<int>     BurstTrie threshold\n";
    cout << "  --c=<int>       QFast bucket size\n";
    cout << "  --Q=<int>       queries per index\n";
    cout << "  --eps=<float>   eps for FQA cgpa equality\n";
    cout << "  --seed=<int>    RNG seed\n\n";
    cout << "GWO-specific opts (when --gwo=1):\n";
    cout << "  --Q=<int>       queries per evaluation (default 800)\n";
    cout << "  --iters=<int>   GWO iterations (default 25)\n";
    cout << "  --swarm=<int>   pack size (wolves) (default 16)\n";
    cout << "  --seed=<int>    RNG seed (default 123)\n";
    cout << "  --kmin=<int> --kmax=<int>     search range for K   (default 4..32)\n";
    cout << "  --thrmin=<int> --thrmax=<int> search range for THR (default 8..256)\n";
    cout << "  --cmin=<int> --cmax=<int>     search range for C   (default 16..256)\n";
    cout << "  --wb=<float>    weight for build time   (default 1.0)\n";
    cout << "  --wq=<float>    weight for query time ms (default 5.0)\n";
    cout << "  --wm=<float>    weight for memory MB     (default 0.001)\n";
    cout << endl;
}

static map<string,string> gwo_parse_args(int argc, char** argv){
    map<string,string> m;
    for (int i=1;i<argc;++i){
        string s(argv[i]);
        if (s.rfind("--",0)==0){
            auto eq = s.find('=');
            if (eq == string::npos) m[s.substr(2)] = "1";
            else m[s.substr(2,eq-2)] = s.substr(eq+1);
        }
    }
    return m;
}

template<class MapT>
static int gwo_geti(const MapT& m, const string& k, int defv){
    auto it = m.find(k);
    if (it==m.end()) return defv;
    return stoi(it->second);
}
template<class MapT>
static double gwo_getd(const MapT& m, const string& k, double defv){
    auto it = m.find(k);
    if (it==m.end()) return defv;
    return stod(it->second);
}
template<class MapT>
static bool gwo_getb(const MapT& m, const string& k, bool defv=false){
    auto it = m.find(k);
    if (it==m.end()) return defv;
    const string& v = it->second;
    return (v=="1" || v=="true" || v=="yes" || v=="on");
}

// -----------------------------------------------------------------------------
// Pretty print result (ABC-style-ish, but labelled for GWO)
// -----------------------------------------------------------------------------
static void print_gwo_result(const BenchConfig& cfg,
                             const BenchResult& res,
                             const CostModel& cm)
{
    cout << "\n================= GWO Optimisation Result =================\n";
    cout << "Best configuration:\n";
    cout << "  K   = " << cfg.K   << "\n";
    cout << "  THR = " << cfg.THR << "\n";
    cout << "  C   = " << cfg.Csize << "\n";

    double total_cost = cm(res);
    cout << "  Cost (with current weights) = " << total_cost << "\n";

    auto ms_to_us = [](double ms){ return ms * 1e3; };
    auto ms_to_ns = [](double ms){ return ms * 1e6; };

    cout << "\nBuild (per build):\n";
    cout << "  FQA:   " << res.fqa_build_ms << " ms  | "
         << ms_to_us(res.fqa_build_ms) << " us  | "
         << ms_to_ns(res.fqa_build_ms) << " ns"
         << "   | resident=" << res.fqa_mem
         << " B   peak(extra)=" << res.fqa_build_peak << " B\n";

    cout << "  BTrie: " << res.bt_build_ms  << " ms  | "
         << ms_to_us(res.bt_build_ms)  << " us  | "
         << ms_to_ns(res.bt_build_ms)  << " ns"
         << "   | resident=" << res.bt_mem
         << " B   peak(extra)=" << res.bt_build_peak << " B\n";

    cout << "  QFast: " << res.qf_build_ms  << " ms  | "
         << ms_to_us(res.qf_build_ms)  << " us  | "
         << ms_to_ns(res.qf_build_ms)  << " ns"
         << "   | resident=" << res.qf_mem
         << " B   peak(extra)=" << res.qf_build_peak << " B\n";

    cout << "\nQuery (per lookup; memory over one full loop):\n";
    cout << "  FQA:   " << res.fqa_q_ms << " ms  | "
         << (res.fqa_q_ms*1e3) << " us  | "
         << (res.fqa_q_ms*1e6) << " ns"
         << "   | loop_peak="        << res.fqa_query_peak
         << " B   loop_total_alloc=" << res.fqa_query_total
         << " B   loop_live_delta="  << res.fqa_query_live_delta << " B\n";

    cout << "  BTrie: " << res.bt_q_ms  << " ms  | "
         << (res.bt_q_ms*1e3)  << " us  | "
         << (res.bt_q_ms*1e6)  << " ns"
         << "   | loop_peak="        << res.bt_query_peak
         << " B   loop_total_alloc=" << res.bt_query_total
         << " B   loop_live_delta="  << res.bt_query_live_delta << " B\n";

    cout << "  QFast: " << res.qf_q_ms  << " ms  | "
         << (res.qf_q_ms*1e3)  << " us  | "
         << (res.qf_q_ms*1e6)  << " ns"
         << "   | loop_peak="        << res.qf_query_peak
         << " B   loop_total_alloc=" << res.qf_query_total
         << " B   loop_live_delta="  << res.qf_query_live_delta << " B\n";

    cout << "===========================================================\n";
}

// -----------------------------------------------------------------------------
// Grey Wolf Optimiser (GWO) engine over (K, THR, Csize)
// -----------------------------------------------------------------------------
struct Wolf {
    array<double,3> x;   // continuous position in [kmin..kmax], [thrmin..thrmax], [cmin..cmax]
    double cost;
};

static inline int clampi(int v, int lo, int hi){
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static BenchConfig discretize_pos(const array<double,3>& x,
                                  int kmin, int kmax,
                                  int tmin, int tmax,
                                  int cmin, int cmax)
{
    BenchConfig cfg;
    cfg.K      = clampi((int)llround(x[0]), kmin, kmax);
    cfg.THR    = clampi((int)llround(x[1]), tmin, tmax);
    cfg.Csize  = clampi((int)llround(x[2]), cmin, cmax);
    return cfg;
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = gwo_parse_args(argc, argv);

    if (args.count("help") || args.count("h")){
        print_usage();
        return 0;
    }

    bool use_gwo = gwo_getb(args, "gwo", false);
    if (!use_gwo){
        cerr << "[INFO] --gwo not set; running baseline_main instead.\n";
        return baseline_main(argc, argv);
    }

    // ------------- Global params for evaluation -------------
    int Q = gwo_geti(args, "Q", 800);          // queries per evaluation
    double EPS = gwo_getd(args, "eps", 1e-9); // cgpa epsilon
    uint32_t seed = (uint32_t)gwo_geti(args, "seed", 123);

    // Search space bounds
    int kmin  = gwo_geti(args, "kmin", 4);
    int kmax  = gwo_geti(args, "kmax", 32);
    int tmin  = gwo_geti(args, "thrmin", 8);
    int tmax  = gwo_geti(args, "thrmax", 256);
    int cmin  = gwo_geti(args, "cmin", 16);
    int cmax  = gwo_geti(args, "cmax", 256);

    if (kmin>kmax) swap(kmin, kmax);
    if (tmin>tmax) swap(tmin, tmax);
    if (cmin>cmax) swap(cmin, cmax);

    int iters     = gwo_geti(args, "iters", 25);
    int pack_size = gwo_geti(args, "swarm", 16); // reuse name 'swarm' for convenience

    // Cost model weights
    CostModel cm;
    cm.w_build = gwo_getd(args, "wb", 1.0);
    cm.w_query = gwo_getd(args, "wq", 5.0);
    cm.w_mem   = gwo_getd(args, "wm", 0.001);

    // Load students via compiled-in dataset
    vector<Student> students = load_students_static();
    if (students.empty()){
        cerr << "Static dataset is empty. Ensure students_data.hpp is included.\n";
        return 1;
    }
    cout << "[GWO] Loaded students: " << students.size() << "\n";
    cout << "[GWO] Q=" << Q << ", iters=" << iters
         << ", pack_size=" << pack_size << ", seed=" << seed << "\n";
    cout << "[GWO] Search ranges: "
         << "K["   << kmin << "," << kmax << "], "
         << "THR[" << tmin << "," << tmax << "], "
         << "C["   << cmin << "," << cmax << "]\n";

    mt19937 rng(seed);
    uniform_real_distribution<double> u01(0.0, 1.0);

    auto rand_in_range = [&](int lo, int hi){
        double r = u01(rng);
        return lo + r*(double)(hi-lo);
    };

    // ------------- Initialise pack of wolves -------------
    vector<Wolf> pack(pack_size);
    BenchConfig best_cfg{};
    BenchResult best_res{};
    double best_cost = numeric_limits<double>::infinity();

    // Alpha, beta, delta wolves according to cost
    Wolf alpha, beta, delta;
    alpha.cost = beta.cost = delta.cost = numeric_limits<double>::infinity();

    for (int i=0;i<pack_size;i++){
        pack[i].x[0] = rand_in_range(kmin, kmax);
        pack[i].x[1] = rand_in_range(tmin, tmax);
        pack[i].x[2] = rand_in_range(cmin, cmax);

        BenchConfig cfg = discretize_pos(pack[i].x, kmin,kmax,tmin,tmax,cmin,cmax);
        BenchResult res = evaluate_config(students, cfg, Q, seed + i*17u, EPS);
        double cost = cm(res);
        pack[i].cost = cost;

        // update global best
        if (cost < best_cost){
            best_cost = cost;
            best_cfg  = cfg;
            best_res  = res;
        }

        // update alpha, beta, delta
        if (cost < alpha.cost){
            delta = beta;
            beta  = alpha;
            alpha = pack[i];
        } else if (cost < beta.cost){
            delta = beta;
            beta  = pack[i];
        } else if (cost < delta.cost){
            delta = pack[i];
        }
    }

    cerr << "[GWO] Initial best: cost=" << best_cost
         << " | K=" << best_cfg.K
         << " THR=" << best_cfg.THR
         << " C=" << best_cfg.Csize << "\n";

    // ------------- Main GWO loop -------------
    for (int iter=0; iter<iters; ++iter){
        double a = 2.0 - 2.0 * (double)iter / max(1, iters-1); // decreases [2 -> 0]

        for (int i=0;i<pack_size;i++){
            for (int d=0; d<3; ++d){
                double Xw = pack[i].x[d];

                double r1 = u01(rng), r2 = u01(rng);
                double A1 = 2.0*a*r1 - a;
                double C1 = 2.0*r2;
                double D_alpha = fabs(C1*alpha.x[d] - Xw);
                double X1 = alpha.x[d] - A1*D_alpha;

                double r3 = u01(rng), r4 = u01(rng);
                double A2 = 2.0*a*r3 - a;
                double C2 = 2.0*r4;
                double D_beta = fabs(C2*beta.x[d] - Xw);
                double X2 = beta.x[d] - A2*D_beta;

                double r5 = u01(rng), r6 = u01(rng);
                double A3 = 2.0*a*r5 - a;
                double C3 = 2.0*r6;
                double D_delta = fabs(C3*delta.x[d] - Xw);
                double X3 = delta.x[d] - A3*D_delta;

                double Xnew = (X1 + X2 + X3) / 3.0;

                // clamp to bounds
                if (d==0){
                    if (Xnew < kmin) Xnew = kmin;
                    if (Xnew > kmax) Xnew = kmax;
                } else if (d==1){
                    if (Xnew < tmin) Xnew = tmin;
                    if (Xnew > tmax) Xnew = tmax;
                } else {
                    if (Xnew < cmin) Xnew = cmin;
                    if (Xnew > cmax) Xnew = cmax;
                }

                pack[i].x[d] = Xnew;
            }

            BenchConfig cfg = discretize_pos(pack[i].x, kmin,kmax,tmin,tmax,cmin,cmax);
            BenchResult res = evaluate_config(students, cfg, Q, seed + 1000u + iter*31u + i, EPS);
            double cost = cm(res);
            pack[i].cost = cost;

            if (cost < best_cost){
                best_cost = cost;
                best_cfg  = cfg;
                best_res  = res;
            }
        }

        // recompute alpha, beta, delta from updated pack
        alpha.cost = beta.cost = delta.cost = numeric_limits<double>::infinity();
        for (int i=0;i<pack_size;i++){
            double cost = pack[i].cost;
            if (cost < alpha.cost){
                delta = beta;
                beta  = alpha;
                alpha = pack[i];
            } else if (cost < beta.cost){
                delta = beta;
                beta  = pack[i];
            } else if (cost < delta.cost){
                delta = pack[i];
            }
        }

        cerr << "[GWO] iter " << (iter+1) << "/" << iters
             << "  best_cost=" << best_cost
             << "  (K=" << best_cfg.K
             << ", THR=" << best_cfg.THR
             << ", C=" << best_cfg.Csize << ")\n";
    }

    // ------------- Final report -------------
    print_gwo_result(best_cfg, best_res, cm);
    return 0;
}
