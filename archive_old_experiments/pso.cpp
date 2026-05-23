// ============================================================================
// pso.cpp — PSO runner that prints results in the SAME format as abc.cpp
// • Place next to: benchmarking_code.cpp (and your dataset headers)
// • Build:  g++ -O2 -std=gnu++17 pso.cpp -o pso
// • Run PSO: ./pso --pso=1 --iters 30 --swarm 16 --Q 3000
// • Run baseline (original program): ./pso
// ============================================================================

#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Include your original TU, but rename its main() to baseline_main().
#define main baseline_main
#include "benchmarking_code.cpp"
#undef main

// ===================== GLOBAL HEAP ALLOCATION TRACKER ======================
namespace memtrack {
    struct Header { size_t size; void* raw; };
    static atomic<size_t> g_live{0};
    static atomic<size_t> g_peak{0};
    static atomic<size_t> g_total{0};

    static inline void note_alloc(size_t sz){
        size_t live = g_live.fetch_add(sz, memory_order_relaxed) + sz;
        g_total.fetch_add(sz, memory_order_relaxed);
        size_t prev = g_peak.load(memory_order_relaxed);
        while (live > prev && !g_peak.compare_exchange_weak(prev, live, memory_order_relaxed)) {}
    }
    static inline void note_free(size_t sz){ g_live.fetch_sub(sz, memory_order_relaxed); }

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
        void reset(){ base_live = g_live.load(); base_total = g_total.load(); base_peak = g_peak.load(); }
        size_t live_delta()  const { size_t cur = g_live.load();  return (cur>=base_live)?  cur-base_live  : 0; }
        size_t total_delta() const { size_t cur = g_total.load(); return (cur>=base_total)? cur-base_total : 0; }
        size_t peak_delta()  const { size_t cur = g_peak.load();  return (cur> base_peak)? cur-base_peak  : 0; }
    };
}
// global new/delete
void* operator new(size_t sz){ return ::memtrack::alloc_block(sz, alignof(max_align_t)); }
void  operator delete(void* p) noexcept { ::memtrack::free_block(p); }
void* operator new[](size_t sz){ return ::memtrack::alloc_block(sz, alignof(max_align_t)); }
void  operator delete[](void* p) noexcept { ::memtrack::free_block(p); }
void  operator delete(void* p, size_t) noexcept { ::memtrack::free_block(p); }
void  operator delete[](void* p, size_t) noexcept { ::memtrack::free_block(p); }
#if (__cpp_aligned_new >= 201606)
void* operator new(size_t sz, std::align_val_t al){ return ::memtrack::alloc_block(sz, (size_t)al); }
void  operator delete(void* p, std::align_val_t) noexcept { ::memtrack::free_block(p); }
void* operator new[](size_t sz, std::align_val_t al){ return ::memtrack::alloc_block(sz, (size_t)al); }
void  operator delete[](void* p, std::align_val_t) noexcept { ::memtrack::free_block(p); }
void  operator delete(void* p, size_t, std::align_val_t) noexcept { ::memtrack::free_block(p); }
void  operator delete[](void* p, size_t, std::align_val_t) noexcept { ::memtrack::free_block(p); }
#endif

// ---------------------- TIMING HELPERS -------------------------------------
using clk = chrono::steady_clock;
static inline long long span_ns(const clk::time_point& a, const clk::time_point& b){
    return chrono::duration_cast<chrono::nanoseconds>(b - a).count();
}
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

// --------------------------- approx_bytes SFINAE ---------------------------
// REPLACE your current approx_bytes SFINAE block with this (non-const friendly)
// This fixes resident memory = 0 when classes expose a non-const approx_bytes().

template<class T>
auto try_approx_bytes(T& t, int) -> decltype(t.approx_bytes(), size_t{}) {
    return t.approx_bytes();        // works for both const and non-const member fn
}
template<class T>
size_t try_approx_bytes(T&, long){
    return 0;                       // fallback when no approx_bytes() exists
}

// ------------------------------ BENCH TYPES --------------------------------
struct BenchConfig { int K; int THR; int Csize; };
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

// ---------------------- QFast helpers (build + find) -----------------------
// Build: prefer build(students, Csize), else build(students)
struct QFBuild {
    template<class T>
    static auto build(T& t, const vector<Student>& s, int C, int) -> decltype(t.build(s, C), void()){
        t.build(s, C);
    }
    template<class T>
    static auto build(T& t, const vector<Student>& s, int, long) -> decltype(t.build(s), void()){
        t.build(s);
    }
};

// ------- countish(): robust result-counter without ambiguity (FIX) ---------
// 1) If T is integral -> return value
template<class T, class = typename enable_if< is_integral<typename decay<T>::type>::value >::type>
static inline size_t countish(T v){ return (size_t)v; }
// 2) If T has .size() -> return size()
template<class T>
static inline auto countish(const T& v) -> decltype(v.size(), size_t{}) { return (size_t)v.size(); }
// 3) Fallback -> treat as 1 result
static inline size_t countish(...) { return 1; }

// Prefer find_reg, else find_exact, else find
template<class QF>
auto qf_try(QF& qf, const string& s, int) -> decltype(qf.find_reg(s)) { return qf.find_reg(s); }
template<class QF>
auto qf_try(QF& qf, const string& s, long) -> decltype(qf.find_exact(s)) { return qf.find_exact(s); }
template<class QF>
auto qf_try(QF& qf, const string& s, char) -> decltype(qf.find(s)) { return qf.find(s); }

template<class QF>
static size_t qf_query_count(QF& qf, const string& s){
    auto r = qf_try(qf, s, 0);
    return countish(r);
}

// ---------------------- EVALUATE ONE CONFIG (time + mem) -------------------
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
            FQAIndex tmp; tmp.build(students, cfg.K);
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
        R.fqa_query_peak = msq.peak_delta();
        R.fqa_query_total = msq.total_delta();
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
        R.bt_query_peak = msq.peak_delta();
        R.bt_query_total = msq.total_delta();
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
        R.qf_query_peak = msq.peak_delta();
        R.qf_query_total = msq.total_delta();
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

// ------------------------------- COST MODEL --------------------------------
struct CostModel {
    // Defaults set to MATCH abc.cpp style in your screenshot:
    double wBuild=1.0, wQuery=5.0, wMem=0.001;
    static inline double bytes_to_mb(size_t b){ return (double)b / (1024.0*1024.0); }
    double operator()(const BenchResult& r) const {
        double build_ms = r.fqa_build_ms + r.bt_build_ms + r.qf_build_ms;
        double query_ms = r.fqa_q_ms + r.bt_q_ms + r.qf_q_ms;
        double mem_mb   = bytes_to_mb(r.fqa_mem + r.bt_mem + r.qf_mem);
        return wBuild*build_ms + wQuery*query_ms + wMem*mem_mb;
    }
};

// ------------------------------- CLI PARSER --------------------------------
static void print_usage(){
    cout << "Usage:\n"
         << "  pso                 : run baseline (your original main)\n"
         << "  pso --pso=1 [opts]  : run PSO optimizer\n"
         << "Options:\n"
         << "  --Q <int>           queries per eval (default 800)\n"
         << "  --iters <int>       PSO iterations (default 25)\n"
         << "  --swarm <int>       swarm size / particles (default 16)\n"
         << "  --seed <uint>       RNG seed (default 1337)\n"
         << "  --kmin/kmax         FQA K bounds (default 4..32)\n"
         << "  --tmin/tmax         BurstTrie threshold bounds (default 8..256)\n"
         << "  --cmin/cmax         QFast bucket size bounds (default 16..256)\n"
         << "  --wbuild <float>    weight for total build time  (default 1.0)\n"
         << "  --wquery <float>    weight for total query time  (default 5.0)\n"
         << "  --wmem   <float>    weight for resident memory   (default 0.001)\n"
         << "  --eps    <float>    equality epsilon for CGPA    (default 1e-9)\n"
         << "Example:\n"
         << "  ./pso --pso=1 --iters 30 --swarm 16 --Q 3000\n";
}

// Avoid name clashes with any parse_args in your baseline TU.
static map<string,string> pso_parse_args(int argc, char** argv){
    map<string,string> m;
    for (int i=1;i<argc;i++){
        string s = argv[i];
        if (s.rfind("--",0)==0){
            s = s.substr(2);
            size_t eq = s.find('=');
            if (eq!=string::npos) m[s.substr(0,eq)] = s.substr(eq+1);
            else {
                if (i+1<argc && string(argv[i+1]).rfind("--",0)!=0) m[s] = argv[++i];
                else m[s] = "1";
            }
        }
    }
    return m;
}
template<class MapT>
static int pso_geti(const MapT& m, const string& k, int dv){
    auto it=m.find(k); if (it==m.end()) return dv; return stoi(it->second);
}
template<class MapT>
static unsigned pso_getu(const MapT& m, const string& k, unsigned dv){
    auto it=m.find(k); if (it==m.end()) return dv; return (unsigned)stoul(it->second);
}
template<class MapT>
static double pso_getd(const MapT& m, const string& k, double dv){
    auto it=m.find(k); if (it==m.end()) return dv; return stod(it->second);
}
template<class MapT>
static bool pso_getb(const MapT& m, const string& k){
    auto it=m.find(k); return (it!=m.end() && it->second!="0");
}

// -------------------------- ABC-STYLE PRINTER (NEW) ------------------------
static void print_abc_style_result(const BenchConfig& cfg, const BenchResult& R, const CostModel& cm){
    auto ms_to_us = [](double ms){ return ms*1000.0; };
    auto ms_to_ns = [](double ms){ return ms*1e6; };

    size_t sum_resident = R.fqa_mem + R.bt_mem + R.qf_mem;
    double sum_build_ms = R.fqa_build_ms + R.bt_build_ms + R.qf_build_ms;
    double sum_query_ms = R.fqa_q_ms + R.bt_q_ms + R.qf_q_ms;

    cout.setf(ios::fixed); cout<<setprecision(9);

    cout << "\n=== PSO Optimization Result ===\n";
    cout << "Best config: K=" << cfg.K
         << "  THR=" << cfg.THR
         << "  Csize=" << cfg.Csize << "\n\n";

    cout << "Build (per build):\n";
    cout << "FQA:   " << R.fqa_build_ms << " ms | "
                      << ms_to_us(R.fqa_build_ms) << " us | "
                      << ms_to_ns(R.fqa_build_ms) << " ns | "
                      << "resident=" << R.fqa_mem << " B  peak(extra)=" << R.fqa_build_peak << " B\n";
    cout << "BTrie: " << R.bt_build_ms << " ms | "
                      << ms_to_us(R.bt_build_ms) << " us | "
                      << ms_to_ns(R.bt_build_ms) << " ns | "
                      << "resident=" << R.bt_mem << " B  peak(extra)=" << R.bt_build_peak << " B\n";
    cout << "QFast: " << R.qf_build_ms << " ms | "
                      << ms_to_us(R.qf_build_ms) << " us | "
                      << ms_to_ns(R.qf_build_ms) << " ns | "
                      << "resident=" << R.qf_mem << " B  peak(extra)=" << R.qf_build_peak << " B\n";
    cout << "Sum time: " << sum_build_ms << " ms\n";
    cout << "Sum resident after build: " << sum_resident << " B\n\n";

    cout << "Query (per lookup time; memory over one full Q-loop):\n";
    cout << "FQA:   " << R.fqa_q_ms << " ms | "
                      << ms_to_us(R.fqa_q_ms) << " us | "
                      << ms_to_ns(R.fqa_q_ms) << " ns | "
                      << "loop_peak=" << R.fqa_query_peak
                      << " B  loop_total_alloc=" << R.fqa_query_total
                      << " B  loop_live_delta=" << R.fqa_query_live_delta << " B\n";
    cout << "BTrie: " << R.bt_q_ms << " ms | "
                      << ms_to_us(R.bt_q_ms) << " us | "
                      << ms_to_ns(R.bt_q_ms) << " ns | "
                      << "loop_peak=" << R.bt_query_peak
                      << " B  loop_total_alloc=" << R.bt_query_total
                      << " B  loop_live_delta=" << R.bt_query_live_delta << " B\n";
    cout << "QFast: " << R.qf_q_ms << " ms | "
                      << ms_to_us(R.qf_q_ms) << " us | "
                      << ms_to_ns(R.qf_q_ms) << " ns | "
                      << "loop_peak=" << R.qf_query_peak
                      << " B  loop_total_alloc=" << R.qf_query_total
                      << " B  loop_live_delta=" << R.qf_query_live_delta << " B\n";
    cout << "Sum time(ms): " << sum_query_ms << "\n\n";

    cout << "objective (w_build=" << cm.wBuild
         << ", w_query=" << cm.wQuery
         << ", w_mem="   << cm.wMem << ") = "
         << (cm(R)) << "\n\n";
}

// ------------------------------- PSO ENGINE --------------------------------
struct Particle {
    array<double,3> x;   // [K, THR, Csize]
    array<double,3> v;
    array<double,3> p;   // personal best
    double bestCost = numeric_limits<double>::infinity();
};
static inline int clampi(int v, int lo, int hi){ return max(lo, min(hi, v)); }
static BenchConfig discretize(const array<double,3>& x, int kmin,int kmax,int tmin,int tmax,int cmin,int cmax){
    BenchConfig bc;
    bc.K     = clampi((int)llround(x[0]), kmin,kmax);
    bc.THR   = clampi((int)llround(x[1]), tmin,tmax);
    bc.Csize = clampi((int)llround(x[2]), cmin,cmax);
    return bc;
}

// --------------------------------- MAIN ------------------------------------
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = pso_parse_args(argc, argv);
    bool run_pso = pso_getb(args, "pso");

    if (!run_pso){
        // Run original program's main (this prints your baseline stats):
        return baseline_main(argc, argv);
    }

    vector<Student> students;
    if (args.count("data")) students = load_students_csv(args["data"]);
    else students = load_students_static();
    
    if (students.empty()){
        cerr << "[ERROR] No students loaded.\n";
        return 2;
    }
    
    int useN = pso_geti(args, "n", -1);
    unsigned seed = pso_getu(args, "seed", 1337u);
    if (useN > 0 && useN < (int)students.size()){
        mt19937 rng(seed);
        shuffle(students.begin(), students.end(), rng);
        students.resize(useN);
    }

    if (students.empty()){
        cerr << "[ERROR] No students loaded. Ensure your data loader (e.g., students_data.hpp) is available.\n";
        return 2;
    }

    // Hyperparameters
    int Q       = pso_geti(args, "Q", 800);
    int iters   = pso_geti(args, "iters", 25);
    int swarmN  = pso_geti(args, "swarm", 16);
    unsigned seed = pso_getu(args, "seed", 1337u);

    int kmin = pso_geti(args, "kmin", 4),   kmax = pso_geti(args, "kmax", 32);
    int tmin = pso_geti(args, "tmin", 8),   tmax = pso_geti(args, "tmax", 256);
    int cmin = pso_geti(args, "cmin", 16),  cmax = pso_geti(args, "cmax", 256);

    CostModel cm;
    cm.wBuild = pso_getd(args, "wbuild", 1.0);
    cm.wQuery = pso_getd(args, "wquery", 5.0);   // match abc default look
    cm.wMem   = pso_getd(args, "wmem",   0.001); // match abc default look
    double EPS = pso_getd(args, "eps", 1e-9);

    // PSO coefficients
    double w_inertia = 0.72;
    double c1 = 1.49, c2 = 1.49;

    // Swarm init
    mt19937 rng(seed);
    uniform_real_distribution<double> u01(0.0, 1.0);
    auto rand_in = [&](int lo, int hi)->double{
        if (hi<=lo) return lo;
        return lo + (hi - lo) * u01(rng);
    };

    vector<Particle> swarm(swarmN);
    for (int i=0;i<swarmN;i++){
        swarm[i].x[0] = rand_in(kmin,kmax);
        swarm[i].x[1] = rand_in(tmin,tmax);
        swarm[i].x[2] = rand_in(cmin,cmax);
        swarm[i].v[0] = swarm[i].v[1] = swarm[i].v[2] = 0.0;
        swarm[i].p = swarm[i].x;
        swarm[i].bestCost = numeric_limits<double>::infinity();
    }

    array<double,3> gbest_x = swarm[0].x;
    double gbest_cost = numeric_limits<double>::infinity();

    cout << "[INFO] PSO enabled -> starting optimizer\n";
    cout << "[PSO] students=" << students.size()
         << " | swarm=" << swarmN
         << " | iters=" << iters
         << " | Q=" << Q << "\n";

    BenchConfig best_cfg{};
    BenchResult best_res{};

    for (int it=1; it<=iters; ++it){
        for (int i=0;i<swarmN;i++){
            BenchConfig cfg = discretize(swarm[i].x, kmin,kmax,tmin,tmax,cmin,cmax);
            BenchResult R = evaluate_config(students, cfg, Q, seed + 12345u + i*17u + it*257u, EPS);
            double cost = cm(R);

            if (cost < swarm[i].bestCost){
                swarm[i].bestCost = cost;
                swarm[i].p = swarm[i].x;
            }
            if (cost < gbest_cost){
                gbest_cost = cost;
                gbest_x = swarm[i].x;
                best_cfg = cfg;
                best_res = R;
            }
        }

        // velocity & position update
        for (int i=0;i<swarmN;i++){
            for (int d=0; d<3; ++d){
                double r1 = u01(rng), r2 = u01(rng);
                swarm[i].v[d] = w_inertia*swarm[i].v[d]
                              + c1*r1*(swarm[i].p[d] - swarm[i].x[d])
                              + c2*r2*(gbest_x[d]    - swarm[i].x[d]);
                swarm[i].x[d] += swarm[i].v[d];
            }
            // clamp to integer design space
            swarm[i].x[0] = clampi((int)llround(swarm[i].x[0]), kmin,kmax);
            swarm[i].x[1] = clampi((int)llround(swarm[i].x[1]), tmin,tmax);
            swarm[i].x[2] = clampi((int)llround(swarm[i].x[2]), cmin,cmax);
        }

        BenchConfig gcfg = discretize(gbest_x, kmin,kmax,tmin,tmax,cmin,cmax);
        cout << "[PSO] iter " << it << "/" << iters
             << " best_cost=" << fixed << setprecision(6) << gbest_cost
             << " | K=" << gcfg.K << " THR=" << gcfg.THR << " Csize=" << gcfg.Csize << "\n";
    }

    // === FINAL: print in EXACT abc.cpp style layout ===
    print_abc_style_result(best_cfg, best_res, cm);
    return 0;
}
