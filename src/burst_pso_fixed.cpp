// ============================================================================
// burst_pso.cpp — Burst Trie + PSO (Particle Swarm Optimization)
// Self-contained single file. Reads CSV directly. No students_data.hpp needed.
//
// Build:  g++ -O2 -std=gnu++17 burst_pso.cpp -o burst_pso
// Run:    ./burst_pso --data=students_100k.csv --n=30000
//
// Tunes Burst Trie threshold THR via PSO, then reports build time, query
// time, and memory for 2 full iterations.
// ============================================================================

#include <bits/stdc++.h>
using namespace std;

// ========================== DATASET RECORD ==================================
struct Student {
    string name;
    string reg_no;
    double cgpa;
};

// ========================== CSV LOADER ======================================
static inline string trim(const string &s) {
    int i = 0, j = (int)s.size() - 1;
    while (i <= j && isspace((unsigned char)s[i])) i++;
    while (j >= i && isspace((unsigned char)s[j])) j--;
    if (i > j) return "";
    return s.substr(i, j - i + 1);
}

static vector<string> split_csv_line(const string &line) {
    vector<string> out;
    string cur;
    bool inq = false;
    for (int i = 0; i < (int)line.size(); i++) {
        char c = line[i];
        if (c == '"') inq = !inq;
        else if (c == ',' && !inq) {
            out.push_back(trim(cur));
            cur.clear();
        } else cur.push_back(c);
    }
    out.push_back(trim(cur));
    return out;
}

static vector<Student> load_students_csv(const string &path) {
    vector<Student> v;
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "ERROR: Cannot open CSV file: " << path << "\n";
        return v;
    }
    string line;
    bool header_skipped = false;
    while (getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto cols = split_csv_line(line);
        if ((int)cols.size() < 3) continue;
        if (!header_skipped) {
            try { (void)stod(cols[2]); }
            catch (...) { header_skipped = true; continue; }
            header_skipped = true;
        }
        Student s;
        s.name = cols[0];
        s.reg_no = cols[1];
        try { s.cgpa = stod(cols[2]); }
        catch (...) { continue; }
        v.push_back(s);
    }
    return v;
}

static string human_bytes(size_t x) {
    static const char *S[] = {"B", "KB", "MB", "GB"};
    int i = 0;
    double d = (double)x;
    while (d >= 1024.0 && i < 3) { d /= 1024.0; i++; }
    ostringstream os;
    os << fixed << setprecision(2) << d << " " << S[i];
    return os.str();
}

// ========================== BURST TRIE (over Name) ==========================
struct BurstTrie {
    struct Node {
        bool isLeaf = true;
        unordered_map<unsigned char, int> nxt;
        vector<int> bucket;
    };

    const vector<Student> *students = nullptr;
    vector<Node> nodes;
    int threshold;

    BurstTrie(int threshold_ = 32) : threshold(max(4, threshold_)) {
        nodes.reserve(4096);
        nodes.push_back(Node());
    }

    void _burst(int u, const vector<int> &ids, size_t depth) {
        // Important: if every key has already ended here, all of them map to the
        // sentinel character 0. Bursting again would recurse forever on the same
        // bucket for duplicate names (e.g. many identical names like "Aarav").
        // In that case we must stop and keep this node as a leaf bucket.
        bool all_ended = true;
        for (int t = 0; t < (int)ids.size(); t++) {
            const string &s = students->at(ids[t]).name;
            if (depth < s.size()) {
                all_ended = false;
                break;
            }
        }
        if (all_ended) {
            nodes[u].isLeaf = true;
            nodes[u].nxt.clear();
            nodes[u].bucket = ids;
            return;
        }

        nodes[u].isLeaf = false;
        nodes[u].nxt.clear();
        nodes[u].bucket.clear();

        for (int t = 0; t < (int)ids.size(); t++) {
            int id = ids[t];
            const string &s = students->at(id).name;
            unsigned char c = (depth < s.size()) ? (unsigned char)s[depth] : (unsigned char)0;

            int v;
            auto it = nodes[u].nxt.find(c);
            if (it == nodes[u].nxt.end()) {
                v = (int)nodes.size();
                nodes.push_back(Node());
                nodes[u].nxt[c] = v;
            } else v = it->second;

            nodes[v].bucket.push_back(id);
        }

        for (auto &kv : nodes[u].nxt) {
            int v = kv.second;
            if ((int)nodes[v].bucket.size() > threshold) {
                auto ids2 = nodes[v].bucket;
                nodes[v].bucket.clear();
                _burst(v, ids2, depth + 1);
            }
        }
    }

    void insert(const vector<Student> &st, int id) {
        if (!students) students = &st;
        int u = 0;
        size_t d = 0;
        const string &s = st[id].name;

        while (true) {
            if (nodes[u].isLeaf) {
                nodes[u].bucket.push_back(id);
                if ((int)nodes[u].bucket.size() > threshold) {
                    auto ids = nodes[u].bucket;
                    nodes[u].bucket.clear();
                    _burst(u, ids, d);
                }
                return;
            } else {
                unsigned char c = (d < s.size()) ? (unsigned char)s[d] : (unsigned char)0;
                auto it = nodes[u].nxt.find(c);
                if (it == nodes[u].nxt.end()) {
                    int v = (int)nodes.size();
                    nodes.push_back(Node());
                    nodes[u].nxt[c] = v;

                    u = v;
                    nodes[u].bucket.push_back(id);
                    if ((int)nodes[u].bucket.size() > threshold) {
                        auto ids = nodes[u].bucket;
                        nodes[u].bucket.clear();
                        _burst(u, ids, d + 1);
                    }
                    return;
                } else {
                    u = it->second;
                    d++;
                }
            }
        }
    }

    void build(const vector<Student> &st, int threshold_ = 32) {
        students = &st;
        threshold = max(4, threshold_);
        nodes.clear();
        nodes.push_back(Node());
        for (int i = 0; i < (int)st.size(); i++) insert(st, i);
    }

    vector<int> find_exact(const string &name) const {
        int u = 0;
        size_t d = 0;

        while (true) {
            if (nodes[u].isLeaf) {
                vector<int> ans;
                for (int i = 0; i < (int)nodes[u].bucket.size(); i++) {
                    int id = nodes[u].bucket[i];
                    if (students->at(id).name == name) ans.push_back(id);
                }
                return ans;
            } else {
                unsigned char c = (d < name.size()) ? (unsigned char)name[d] : (unsigned char)0;
                auto it = nodes[u].nxt.find(c);
                if (it == nodes[u].nxt.end()) return {};
                u = it->second;
                d++;
            }
        }
    }

    static size_t approx_unordered_map_bytes(const unordered_map<unsigned char, int> &m) {
        size_t pairs = m.size() * sizeof(pair<const unsigned char, int>);
        size_t buckets = m.bucket_count() * sizeof(void *);
        return pairs + buckets;
    }

    size_t approx_bytes() const {
        size_t bytes = sizeof(Node) * nodes.capacity();
        for (int i = 0; i < (int)nodes.size(); i++) {
            bytes += nodes[i].bucket.capacity() * sizeof(int);
            if (!nodes[i].isLeaf) bytes += approx_unordered_map_bytes(nodes[i].nxt);
        }
        return bytes;
    }
};

// ========================== TIMING HELPERS ===================================
using clk = chrono::steady_clock;

static inline double elapsed_ms(const clk::time_point &a, const clk::time_point &b) {
    return chrono::duration<double, milli>(b - a).count();
}

static inline void do_not_optimize_away(const volatile void *p) {
#if defined(__GNUG__) || defined(__clang__)
    asm volatile("" : : "g"(p) : "memory");
#else
    atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// ========================== EVALUATION FUNCTION ==============================
struct EvalResult {
    double build_ms;
    double query_ms;
    size_t mem_bytes;
};

static EvalResult evaluate_burst(const vector<Student> &students, int THR,
                                 int Q, uint32_t seed, double EPS)
{
    EvalResult er{};
    static volatile size_t sink = 0;

    BurstTrie bt;
    auto t0 = clk::now();
    bt.build(students, THR);
    auto t1 = clk::now();
    er.build_ms = elapsed_ms(t0, t1);

    er.mem_bytes = bt.approx_bytes();

    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, (int)students.size() - 1);
    vector<string> name_queries(Q);
    for (int i = 0; i < Q; i++) {
        name_queries[i] = students[dist(rng)].name;
    }

    auto q0 = clk::now();
    for (int i = 0; i < Q; i++) {
        auto res = bt.find_exact(name_queries[i]);
        sink += res.size();
    }
    auto q1 = clk::now();
    er.query_ms = elapsed_ms(q0, q1);

    do_not_optimize_away(&sink);
    return er;
}

// ========================== PSO OPTIMIZER ====================================
struct Particle {
    double x;
    double v;
    double pbest_x;
    double pbest_cost;
};

struct PSOOptimizer {
    int THRmin = 8, THRmax = 256;

    int swarm_size = 16;
    int max_iters  = 25;

    double w_inertia = 0.72;
    double c1 = 1.49;
    double c2 = 1.49;

    double w_build = 1.0;
    double w_query = 1.0;
    double w_mem   = 0.001;

    const vector<Student> *studs = nullptr;
    int Q = 2000;
    uint32_t seed = 42u;
    double EPS = 1e-9;

    mt19937 rng{1337};

    static int clampi(int v, int lo, int hi) { return max(lo, min(hi, v)); }

    double compute_cost(const EvalResult &r) const {
        double mem_mb = (double)r.mem_bytes / (1024.0 * 1024.0);
        return w_build * r.build_ms + w_query * r.query_ms + w_mem * mem_mb;
    }

    int to_THR(double pos) const {
        return clampi((int)llround(pos), THRmin, THRmax);
    }

    struct PSOResult {
        int best_THR;
        EvalResult best_res;
        double best_cost;
    };

    PSOResult optimize() {
        uniform_real_distribution<double> u01(0.0, 1.0);

        auto rand_pos = [&]() -> double {
            return THRmin + (THRmax - THRmin) * u01(rng);
        };

        vector<Particle> swarm(swarm_size);
        for (int i = 0; i < swarm_size; i++) {
            swarm[i].x = rand_pos();
            swarm[i].v = 0.0;
            swarm[i].pbest_x = swarm[i].x;
            swarm[i].pbest_cost = numeric_limits<double>::infinity();
        }

        double gbest_x = swarm[0].x;
        double gbest_cost = numeric_limits<double>::infinity();
        int gbest_THR = to_THR(gbest_x);
        EvalResult gbest_res{};

        for (int i = 0; i < swarm_size; i++) {
            int THR = to_THR(swarm[i].x);
            EvalResult res = evaluate_burst(*studs, THR, Q,
                seed + (uint32_t)(i * 17), EPS);
            double cost = compute_cost(res);

            swarm[i].pbest_cost = cost;
            swarm[i].pbest_x = swarm[i].x;

            if (cost < gbest_cost) {
                gbest_cost = cost;
                gbest_x = swarm[i].x;
                gbest_THR = THR;
                gbest_res = res;
            }
        }

        cerr << "[PSO] Initial best: cost=" << fixed << setprecision(6)
             << gbest_cost << " THR=" << gbest_THR << "\n";

        for (int it = 1; it <= max_iters; it++) {
            for (int i = 0; i < swarm_size; i++) {
                double r1 = u01(rng);
                double r2 = u01(rng);
                swarm[i].v = w_inertia * swarm[i].v
                           + c1 * r1 * (swarm[i].pbest_x - swarm[i].x)
                           + c2 * r2 * (gbest_x - swarm[i].x);

                swarm[i].x += swarm[i].v;

                if (swarm[i].x < THRmin) swarm[i].x = THRmin;
                if (swarm[i].x > THRmax) swarm[i].x = THRmax;

                int THR = to_THR(swarm[i].x);
                EvalResult res = evaluate_burst(*studs, THR, Q,
                    seed + (uint32_t)(12345 + it * 257 + i * 17), EPS);
                double cost = compute_cost(res);

                if (cost < swarm[i].pbest_cost) {
                    swarm[i].pbest_cost = cost;
                    swarm[i].pbest_x = swarm[i].x;
                }

                if (cost < gbest_cost) {
                    gbest_cost = cost;
                    gbest_x = swarm[i].x;
                    gbest_THR = THR;
                    gbest_res = res;
                }
            }

            cerr << "[PSO] iter " << it << "/" << max_iters
                 << "  best_cost=" << fixed << setprecision(6) << gbest_cost
                 << "  THR=" << gbest_THR << "\n";
        }

        return PSOResult{gbest_THR, gbest_res, gbest_cost};
    }
};

// ========================== CLI PARSER =======================================
static map<string, string> parse_args(int argc, char **argv) {
    map<string, string> m;
    for (int i = 1; i < argc; i++) {
        string s(argv[i]);
        if (s.rfind("--", 0) == 0) {
            auto eq = s.find('=');
            if (eq == string::npos) {
                if (i + 1 < argc && string(argv[i + 1]).rfind("--", 0) != 0)
                    m[s.substr(2)] = argv[++i];
                else
                    m[s.substr(2)] = "1";
            } else {
                m[s.substr(2, eq - 2)] = s.substr(eq + 1);
            }
        }
    }
    return m;
}

// ========================== MAIN =============================================
int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);

    string data_path = args.count("data") ? args["data"] : "students_100k.csv";
    int N = args.count("n") ? stoi(args["n"]) : 30000;
    uint32_t seed = args.count("seed") ? (uint32_t)stoul(args["seed"]) : 42u;
    int Q = args.count("Q") ? stoi(args["Q"]) : 2000;
    double EPS = args.count("eps") ? stod(args["eps"]) : 1e-9;

    int tmin   = args.count("tmin")   ? stoi(args["tmin"])   : 8;
    int tmax   = args.count("tmax")   ? stoi(args["tmax"])   : 256;
    int swarm  = args.count("swarm")  ? stoi(args["swarm"])  : 16;
    int iters  = args.count("iters")  ? stoi(args["iters"])  : 25;

    int NUM_ITERATIONS = 2;

    cout << "============================================================\n";
    cout << "  Burst Trie + PSO Optimization (30k dataset)\n";
    cout << "============================================================\n\n";

    vector<Student> all_students = load_students_csv(data_path);
    if (all_students.empty()) {
        cerr << "ERROR: Could not load dataset from: " << data_path << "\n";
        return 1;
    }
    cout << "Loaded " << all_students.size() << " students from: " << data_path << "\n";

    if (N > 0 && N < (int)all_students.size()) {
        mt19937 rng_shuffle(seed);
        shuffle(all_students.begin(), all_students.end(), rng_shuffle);
        all_students.resize(N);
    }
    cout << "Using subset: N = " << all_students.size() << "\n";
    cout << "Queries per evaluation: Q = " << Q << "\n";
    cout << "PSO params: swarm_size=" << swarm << ", max_iters=" << iters
         << ", THR range=[" << tmin << "," << tmax << "]\n";
    cout << "PSO coefficients: w_inertia=0.72, c1=1.49, c2=1.49\n";
    cout << "Number of iterations: " << NUM_ITERATIONS << "\n\n";

    for (int iter = 1; iter <= NUM_ITERATIONS; iter++) {
        cout << "------------------------------------------------------------\n";
        cout << "  ITERATION " << iter << " of " << NUM_ITERATIONS << "\n";
        cout << "------------------------------------------------------------\n";

        PSOOptimizer pso;
        pso.THRmin = tmin;
        pso.THRmax = tmax;
        pso.swarm_size = swarm;
        pso.max_iters = iters;
        pso.studs = &all_students;
        pso.Q = Q;
        pso.seed = seed + (uint32_t)(iter * 7919);
        pso.EPS = EPS;
        pso.rng.seed(seed + iter * 31337u);

        auto t_opt_start = clk::now();
        auto best = pso.optimize();
        auto t_opt_end = clk::now();
        double opt_time_ms = elapsed_ms(t_opt_start, t_opt_end);

        cout << "\n";
        cout << "  PSO Optimization complete (took " << fixed << setprecision(2)
             << opt_time_ms << " ms)\n";
        cout << "  Best THR (threshold) found: " << best.best_THR << "\n\n";

        cout << "  --- Final Evaluation with optimized THR=" << best.best_THR << " ---\n";

        BurstTrie bt;
        auto tb0 = clk::now();
        bt.build(all_students, best.best_THR);
        auto tb1 = clk::now();
        double final_build_ms = elapsed_ms(tb0, tb1);

        size_t final_mem = bt.approx_bytes();

        mt19937 qrng(seed + iter * 9973u);
        uniform_int_distribution<int> qdist(0, (int)all_students.size() - 1);
        vector<string> name_queries(Q);
        for (int i = 0; i < Q; i++) {
            name_queries[i] = all_students[qdist(qrng)].name;
        }

        volatile size_t sink = 0;
        auto tq0 = clk::now();
        for (int i = 0; i < Q; i++) {
            auto res = bt.find_exact(name_queries[i]);
            sink += res.size();
        }
        auto tq1 = clk::now();
        double final_query_ms = elapsed_ms(tq0, tq1);
        do_not_optimize_away(&sink);

        cout << "\n";
        cout << "  +-------------------------------------------------+\n";
        cout << "  | Burst + PSO |  Iteration " << iter << "                     |\n";
        cout << "  +-------------------------------------------------+\n";
        cout << "  | Optimized THR (threshold): " << best.best_THR << "\n";
        cout << "  | Build Time            : " << fixed << setprecision(6)
             << final_build_ms << " ms\n";
        cout << "  | Query Time (" << Q << " queries): " << fixed << setprecision(6)
             << final_query_ms << " ms\n";
        cout << "  | Avg Query Time        : " << fixed << setprecision(6)
             << (final_query_ms / Q) << " ms/query\n";
        cout << "  | Memory (approx)       : " << final_mem << " bytes ("
             << human_bytes(final_mem) << ")\n";
        cout << "  +-------------------------------------------------+\n";
        cout << "\n";
    }

    cout << "============================================================\n";
    cout << "  Burst Trie + PSO complete. " << NUM_ITERATIONS << " iterations done.\n";
    cout << "============================================================\n";

    return 0;
}
