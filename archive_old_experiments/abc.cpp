// ============================================================================
// abc.cpp — Single-file ABC runner with detailed TIME + MEMORY (build & query)
// • Put this file next to: benchmarking_code.cpp  and  students_data.hpp
// • Build:  g++ -O2 -std=gnu++17 abc.cpp -o abc
// • Run ABC:  ./abc --abc=1 --cycles 20 --sn 12 --Q 3000
// • Run baseline:  ./abc
// ============================================================================

#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Include your whole project after renaming main(), so baseline still works.
#define main baseline_main
#include "benchmarking_code.cpp"   // defines Student, FQAIndex, BurstTrie, QFastTrie, load_students_static()
#undef main

static void print_usage(){
    cout << "Usage:\n"
         << "  abc                 : run baseline (your original main)\n"
         << "  abc --abc=1 [opts]  : run ABC optimizer\n"
         << "Options:\n"
         << "  --Q <int>           queries per eval (default 800)\n"
         << "  --sn <int>          food sources (default 12)\n"
         << "  --limit <int>       scout abandon threshold (default 8)\n"
         << "  --cycles <int>      ABC iterations (default 25)\n"
         << "  --kmin/kmax         FQA K bounds (default 4..32)\n"
         << "  --tmin/tmax         BurstTrie threshold bounds (8..256)\n"
         << "  --cmin/cmax         QFast bucket size bounds (16..256)\n"
         << "  --wbuild --wquery --wmem  objective weights\n"
         << "Example:\n"
         << "  ./abc --abc=1 --cycles 20 --sn 12 --Q 3000\n";
}

// ========================= ABC Optimizer (self-contained) =========================
namespace abcopt {

// --- tiny CLI parser (isolated) ---
static unordered_map<string,string> parse_args_abc(int argc, char** argv){
    unordered_map<string,string> m;
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

} // namespace abcopt

// ===================== GLOBAL HEAP ALLOCATION TRACKER ======================
// We override global new/delete to count bytes. This lets us report:
//  • peak extra bytes during build/query
//  • total bytes allocated during query loop
//  • live bytes delta (should be ~0 after a query loop)
// NOTE: these must be at global scope (NOT inside a namespace).

namespace memtrack {
    struct Header { size_t size; void* raw; };

    static atomic<size_t> g_live{0};    // current live bytes
    static atomic<size_t> g_peak{0};    // high-water mark of live bytes
    static atomic<size_t> g_total{0};   // cumulative allocated bytes

    static inline void note_alloc(size_t sz){
        size_t live = g_live.fetch_add(sz, memory_order_relaxed) + sz;
        g_total.fetch_add(sz, memory_order_relaxed);
        size_t prev = g_peak.load(memory_order_relaxed);
        while (live > prev && !g_peak.compare_exchange_weak(prev, live, memory_order_relaxed)) {}
    }
    static inline void note_free(size_t sz){
        g_live.fetch_sub(sz, memory_order_relaxed);
    }

    // align helper
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
        Scope(){ base_live = g_live.load(); base_total = g_total.load(); base_peak = g_peak.load(); }
        void reset(){ base_live = g_live.load(); base_total = g_total.load(); base_peak = g_peak.load(); }
        size_t live_delta() const { size_t cur = g_live.load(); return (cur>=base_live)? cur-base_live : 0; }
        size_t total_delta() const { size_t cur = g_total.load(); return (cur>=base_total)? cur-base_total : 0; }
        size_t peak_delta() const {
            size_t cur = g_peak.load();
            return (cur>base_peak)? (cur-base_peak) : 0;
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

// ---------------------- TIMING + ANTI-OPT HELPERS --------------------------
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

// --------------------------- approx_bytes SFINAE ---------------------------
template<class T>
auto try_approx_bytes(const T& t, int) -> decltype(t.approx_bytes(), size_t{}){
    return t.approx_bytes();
}
template<class T>
size_t try_approx_bytes(const T&, long){ return 0; }

// ------------------------------ BENCH TYPES --------------------------------
struct BenchConfig { int K; int THR; int Csize; };
struct BenchResult {
    // time
    double build_ms_sum=0.0, query_ms_sum=0.0;
    double fqa_build_ms=0.0, bt_build_ms=0.0, qf_build_ms=0.0;
    double fqa_q_ms=0.0,     bt_q_ms=0.0,    qf_q_ms=0.0;
    // resident memory after build (structure size)
    size_t fqa_mem=0,        bt_mem=0,       qf_mem=0;
    size_t mem_bytes_sum=0;

    // nano timing (detail)
    double fqa_build_ns=0.0, bt_build_ns=0.0, qf_build_ns=0.0;
    double fqa_q_ns=0.0,     bt_q_ns=0.0,     qf_q_ns=0.0; // per-query

    // NEW: memory during build (peak extra) and during queries (peak/total/live_delta)
    size_t fqa_build_peak=0, bt_build_peak=0, qf_build_peak=0; // extra bytes peak while building
    size_t fqa_query_peak=0, bt_query_peak=0, qf_query_peak=0; // extra bytes peak while querying (one loop)
    size_t fqa_query_total=0, bt_query_total=0, qf_query_total=0; // cumulative alloc bytes during one loop
    size_t fqa_query_live_delta=0, bt_query_live_delta=0, qf_query_live_delta=0; // ~0 => no leak
};

// ---------------------- EVALUATE ONE CONFIG (time + mem) -------------------
static BenchResult evaluate_config(const vector<Student>& students,
                                   const BenchConfig& cfg,
                                   int Q, uint32_t seed, double EPS)
{
    BenchResult R{};
    static volatile size_t sink_guard = 0; // touch results to avoid DCE

    // ---------- FQA (build + queries) ----------
    {
        // Build memory (peak) + resident
        memtrack::Scope msb;
        FQAIndex fqa;
        fqa.build(students, cfg.K);
        R.fqa_mem = try_approx_bytes(fqa, 0);
        R.fqa_build_peak = msb.peak_delta(); // transient peak during build

        // Build time (average over many independent builds)
        int build_reps = 20;
        auto build_ns = avg_ns_runs([&](){
            FQAIndex tmp;
            tmp.build(students, cfg.K);
            sink_guard ^= try_approx_bytes(tmp, 0) + 1;
            do_not_optimize_away(&tmp);
        }, build_reps);
        R.fqa_build_ns = (double)build_ns;
        R.fqa_build_ms = (double)build_ns / 1e6;

        // Prepare queries
        mt19937 rng(seed);
        uniform_int_distribution<int> dist(0, (int)students.size()-1);
        vector<double> cg_queries(Q);
        for (int i=0;i<Q;i++) cg_queries[i] = students[ dist(rng) ].cgpa;

        // Query memory on ONE full loop
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
        R.fqa_query_live_delta = msq.live_delta(); // ~0 expected

        // Query time (avg multiple loops)
        int loop_reps = 100;
        double ns_full = avg_loop_ns(one_loop, loop_reps);
        double ns_per_query = ns_full / max(1, Q);
        R.fqa_q_ns = ns_per_query;
        R.fqa_q_ms = ns_per_query / 1e6;
        sink_guard ^= (size_t)R.fqa_q_ns;
    }

    // ---------- BurstTrie (build + queries) ----------
    {
        memtrack::Scope msb;
        BurstTrie bt;
        bt.build(students, cfg.THR);
        R.bt_mem = try_approx_bytes(bt, 0);
        R.bt_build_peak = msb.peak_delta();

        int build_reps = 20;
        auto build_ns = avg_ns_runs([&](){
            BurstTrie tmp;
            tmp.build(students, cfg.THR);
            sink_guard ^= try_approx_bytes(tmp, 0) + 1;
            do_not_optimize_away(&tmp);
        }, build_reps);
        R.bt_build_ns = (double)build_ns;
        R.bt_build_ms = (double)build_ns / 1e6;

        mt19937 rng(seed+1);
        uniform_int_distribution<int> dist(0, (int)students.size()-1);
        vector<string> name_queries(Q);
        for (int i=0;i<Q;i++) name_queries[i] = students[ dist(rng) ].name;

        auto one_loop = [&](){
            for (int i=0;i<Q;i++){
                volatile auto res = bt.find_exact(name_queries[i]); // vector<int>
                (void)res;
            }
        };
        memtrack::Scope msq;
        one_loop();
        R.bt_query_peak = msq.peak_delta();
        R.bt_query_total = msq.total_delta();
        R.bt_query_live_delta = msq.live_delta();

        int loop_reps = 100;
        double ns_full = avg_loop_ns(one_loop, loop_reps);
        double ns_per_query = ns_full / max(1, Q);
        R.bt_q_ns = ns_per_query;
        R.bt_q_ms = ns_per_query / 1e6;
        sink_guard ^= (size_t)R.bt_q_ns;
    }

    // ---------- QFastTrie (build + queries) ----------
    {
        memtrack::Scope msb;
        QFastTrie qf;
        qf.build(students, cfg.Csize);
        R.qf_mem = try_approx_bytes(qf, 0);
        R.qf_build_peak = msb.peak_delta();

        int build_reps = 20;
        auto build_ns = avg_ns_runs([&](){
            QFastTrie tmp;
            tmp.build(students, cfg.Csize);
            sink_guard ^= try_approx_bytes(tmp, 0) + 1;
            do_not_optimize_away(&tmp);
        }, build_reps);
        R.qf_build_ns = (double)build_ns;
        R.qf_build_ms = (double)build_ns / 1e6;

        mt19937 rng(seed+2);
        uniform_int_distribution<int> dist(0, (int)students.size()-1);
        vector<string> reg_queries(Q);
        for (int i=0;i<Q;i++) reg_queries[i] = students[ dist(rng) ].reg_no;

        auto one_loop = [&](){
            for (int i=0;i<Q;i++){
                volatile int idx = qf.find_exact(reg_queries[i]); // int
                (void)idx;
            }
        };
        memtrack::Scope msq;
        one_loop();
        R.qf_query_peak = msq.peak_delta();
        R.qf_query_total = msq.total_delta();
        R.qf_query_live_delta = msq.live_delta();

        int loop_reps = 100;
        double ns_full = avg_loop_ns(one_loop, loop_reps);
        double ns_per_query = ns_full / max(1, Q);
        R.qf_q_ns = ns_per_query;
        R.qf_q_ms = ns_per_query / 1e6;
        sink_guard ^= (size_t)R.qf_q_ns;
    }

    // Sums (ms / resident bytes after build)
    R.build_ms_sum  = R.fqa_build_ms + R.bt_build_ms + R.qf_build_ms;
    R.query_ms_sum  = R.fqa_q_ms     + R.bt_q_ms     + R.qf_q_ms;
    R.mem_bytes_sum = R.fqa_mem + R.bt_mem + R.qf_mem;

    do_not_optimize_away(&sink_guard);
    return R;
}

// --------------------------- OBJECTIVE FUNCTION ----------------------------
static double fitness_cost(const BenchResult& R, double w_build, double w_query, double w_mem){
    double mem_mb = (double)R.mem_bytes_sum / (1024.0*1024.0);
    return w_build*R.build_ms_sum + w_query*R.query_ms_sum + w_mem*mem_mb;
}

// --------------------------------- ABC -------------------------------------
namespace abcopt {

struct ABC {
    // bounds
    int Kmin=4,   Kmax=32;
    int THmin=8,  THmax=256;
    int Cmin=16,  Cmax=256;

    // ABC hyperparams
    int SN=12, limit=8, max_cycles=25;

    // objective weights
    double w_build=1.0, w_query=5.0, w_mem=0.001;

    // context
    const vector<Student>* studs=nullptr;
    int Q=800; uint32_t seed=42u; double EPS=1e-9;

    struct Food {
        BenchConfig x{};
        BenchResult res{};
        double cost = numeric_limits<double>::infinity();
        int trials = 0;
    };

    mt19937 rng{1234567};
    int rnd_int(int a,int b){ uniform_int_distribution<int> d(a,b); return d(rng); }
    double rnd(){ uniform_real_distribution<double> d(0.0,1.0); return d(rng); }
    double rnd_sym(){ return rnd()*2.0-1.0; }
    static int clampi(int v,int lo,int hi){ return max(lo,min(hi,v)); }

    Food random_food(){
        Food f;
        f.x.K     = rnd_int(Kmin,Kmax);
        f.x.THR   = rnd_int(THmin,THmax);
        f.x.Csize = rnd_int(Cmin,Cmax);
        f.res = evaluate_config(*studs, f.x, Q, seed, EPS);
        f.cost = fitness_cost(f.res, w_build, w_query, w_mem);
        return f;
    }
    void mutate_neighbor(Food& out, const Food& base, const Food& other){
        auto mk=[&](int a,int b,int lo,int hi){
            double phi=rnd_sym();
            double v=a + phi*(a-b);
            return clampi((int)llround(v), lo, hi);
        };
        out=base;
        out.x.K     = mk(base.x.K,     other.x.K,     Kmin,Kmax);
        out.x.THR   = mk(base.x.THR,   other.x.THR,   THmin,THmax);
        out.x.Csize = mk(base.x.Csize, other.x.Csize, Cmin,Cmax);
        out.res = evaluate_config(*studs, out.x, Q, seed, EPS);
        out.cost = fitness_cost(out.res, w_build, w_query, w_mem);
    }

    Food optimize(){
        vector<Food> foods; foods.reserve(SN);
        for (int i=0;i<SN;i++) foods.push_back(random_food());
        Food gbest = *min_element(foods.begin(), foods.end(),
                                  [](const Food& a,const Food& b){ return a.cost<b.cost; });

        for (int cy=0; cy<max_cycles; ++cy){
            // Employed
            for (int i=0;i<SN;i++){
                int j=i; while (j==i) j=rnd_int(0,SN-1);
                Food cand; mutate_neighbor(cand, foods[i], foods[j]);
                if (cand.cost < foods[i].cost){ foods[i]=cand; foods[i].trials=0; if (cand.cost<gbest.cost) gbest=cand; }
                else foods[i].trials++;
            }
            // Onlookers
            vector<double> q(SN); double sumq=0.0;
            for (int i=0;i<SN;i++){ q[i]=1.0/(1.0+foods[i].cost); sumq+=q[i]; }
            for (int r=0;r<SN;r++){
                double pick=rnd()*sumq, acc=0.0; int i=0;
                for(; i<SN; ++i){ acc+=q[i]; if (acc>=pick) break; }
                if (i>=SN) i=SN-1;
                int j=i; while (j==i) j=rnd_int(0,SN-1);
                Food cand; mutate_neighbor(cand, foods[i], foods[j]);
                if (cand.cost < foods[i].cost){ foods[i]=cand; foods[i].trials=0; if (cand.cost<gbest.cost) gbest=cand; }
                else foods[i].trials++;
            }
            // Scouts
            for (int i=0;i<SN;i++){
                if (foods[i].trials > limit){
                    foods[i]=random_food();
                    if (foods[i].cost < gbest.cost) gbest=foods[i];
                }
            }
            cerr<<"[ABC] cycle "<<(cy+1)<<"/"<<max_cycles
                <<" best_cost="<<gbest.cost
                <<" | K="<<gbest.x.K<<" THR="<<gbest.x.THR<<" C="<<gbest.x.Csize<<"\n";
        }
        return gbest;
    }
};

} // namespace abcopt

// =============================== New main() ===============================
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = abcopt::parse_args_abc(argc, argv);

    if (args.count("help") || args.count("h")) {
        print_usage();
        return 0;
    }

    bool runABC = (args.count("abc") && stoi(args["abc"])!=0);
    if (!runABC){
        cerr << "[INFO] ABC disabled -> running baseline (original main)\n";
        return baseline_main(argc, argv);
    }

    cerr << "[INFO] ABC enabled -> starting optimizer\n";

    // --- ABC settings with defaults (override via flags) ---
    abcopt::ABC abc;
    if (args.count("Q"))      abc.Q       = stoi(args["Q"]);
    if (args.count("seed"))   abc.seed    = (uint32_t)stoul(args["seed"]);
    if (args.count("eps"))    abc.EPS     = stod(args["eps"]);
    if (args.count("kmin"))   abc.Kmin    = stoi(args["kmin"]);
    if (args.count("kmax"))   abc.Kmax    = stoi(args["kmax"]);
    if (args.count("tmin"))   abc.THmin   = stoi(args["tmin"]);
    if (args.count("tmax"))   abc.THmax   = stoi(args["tmax"]);
    if (args.count("cmin"))   abc.Cmin    = stoi(args["cmin"]);
    if (args.count("cmax"))   abc.Cmax    = stoi(args["cmax"]);
    if (args.count("sn"))     abc.SN      = stoi(args["sn"]);
    if (args.count("limit"))  abc.limit   = stoi(args["limit"]);
    if (args.count("cycles")) abc.max_cycles = stoi(args["cycles"]);
    if (args.count("wbuild")) abc.w_build = stod(args["wbuild"]);
    if (args.count("wquery")) abc.w_query = stod(args["wquery"]);
    if (args.count("wmem"))   abc.w_mem   = stod(args["wmem"]);

    vector<Student> students;
    if (args.count("data")) students = load_students_csv(args["data"]);
    else students = load_students_static();
    
    if (students.empty()){
        cerr << "[ABC] ERROR: Dataset is empty.\n";
        return 1;
    }
    
    // optional subset sizing (deterministic)
    int useN = -1;
    if (args.count("n")) useN = stoi(args["n"]);
    if (useN > 0 && useN < (int)students.size()){
        mt19937 rng(abc.seed);
        shuffle(students.begin(), students.end(), rng);
        students.resize(useN);
    }
    
    abc.studs = &students;
    cerr << "[ABC] students=" << students.size() << "\n";


    // ------- Build: time + memory (resident + peak transient) -------
    cout << "\nBuild (per build):\n";
    cout << "  FQA:   " << best.res.fqa_build_ms << " ms  | "
         << ms_to_us(best.res.fqa_build_ms) << " us  | "
         << ms_to_ns(best.res.fqa_build_ms) << " ns"
         << "   | resident=" << best.res.fqa_mem
         << " B   peak(extra)=" << best.res.fqa_build_peak << " B\n";
    cout << "  BTrie: " << best.res.bt_build_ms  << " ms  | "
         << ms_to_us(best.res.bt_build_ms)  << " us  | "
         << ms_to_ns(best.res.bt_build_ms)  << " ns"
         << "   | resident=" << best.res.bt_mem
         << " B   peak(extra)=" << best.res.bt_build_peak << " B\n";
    cout << "  QFast: " << best.res.qf_build_ms  << " ms  | "
         << ms_to_us(best.res.qf_build_ms)  << " us  | "
         << ms_to_ns(best.res.qf_build_ms)  << " ns"
         << "   | resident=" << best.res.qf_mem
         << " B   peak(extra)=" << best.res.qf_build_peak << " B\n";
    cout << "  Sum time:   " << best.res.build_ms_sum << " ms\n";
    cout << "  Sum resident after build: " << (best.res.mem_bytes_sum) << " B\n";

    // ------- Query: per-lookup time + memory during one full loop -------
    cout << "\nQuery (per lookup time; memory over one full Q-loop):\n";
    cout << "  FQA:   " << best.res.fqa_q_ms << " ms  | "
         << (best.res.fqa_q_ms*1e3) << " us  | "
         << (best.res.fqa_q_ms*1e6) << " ns"
         << "   | loop_peak=" << best.res.fqa_query_peak
         << " B   loop_total_alloc=" << best.res.fqa_query_total
         << " B   loop_live_delta=" << best.res.fqa_query_live_delta << " B\n";
    cout << "  BTrie: " << best.res.bt_q_ms  << " ms  | "
         << (best.res.bt_q_ms*1e3)  << " us  | "
         << (best.res.bt_q_ms*1e6)  << " ns"
         << "   | loop_peak=" << best.res.bt_query_peak
         << " B   loop_total_alloc=" << best.res.bt_query_total
         << " B   loop_live_delta=" << best.res.bt_query_live_delta << " B\n";
    cout << "  QFast: " << best.res.qf_q_ms  << " ms  | "
         << (best.res.qf_q_ms*1e3)  << " us  | "
         << (best.res.qf_q_ms*1e6)  << " ns"
         << "   | loop_peak=" << best.res.qf_query_peak
         << " B   loop_total_alloc=" << best.res.qf_query_total
         << " B   loop_live_delta=" << best.res.qf_query_live_delta << " B\n";
    cout << "  Sum time(ms): " << best.res.query_ms_sum << "\n";

    // Cost
    cout << "\nObjective (w_build=" << abc.w_build
         << ", w_query=" << abc.w_query
         << ", w_mem="   << abc.w_mem   << ") = "
         << fitness_cost(best.res, abc.w_build, abc.w_query, abc.w_mem) << "\n";

    return 0;
}
