// ============================================================================
// burst_gwo.cpp — Burst Trie + GWO (Grey Wolf Optimizer)
// Self-contained single file. Reads CSV directly. No students_data.hpp needed.
//
// Build:  g++ -O2 -std=gnu++17 burst_gwo.cpp -o burst_gwo
// Run:    ./burst_gwo --data=students_100k.csv --n=30000
//
// Tunes Burst Trie threshold THR via GWO, then reports build time, query
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

// ========================== GWO OPTIMIZER ====================================
struct Wolf {
    double x;
    double cost;
};

struct GWOOptimizer {
    int THRmin = 8, THRmax = 256;
    int pack_size = 16;
    int max_iters = 25;

    double w_build = 1.0;
    double w_query = 1.0;
    double w_mem   = 0.001;

    const vector<Student> *studs = nullptr;
    int Q = 2000;
    uint32_t seed = 42u;
    double EPS = 1e-9;

    mt19937 rng{123};

    static int clampi(int v, int lo, int hi) { return max(lo, min(hi, v)); }

    double compute_cost(const EvalResult &r) const {
        double mem_mb = (double)r.mem_bytes / (1024.0 * 1024.0);
        return w_build * r.build_ms + w_query * r.query_ms + w_mem * mem_mb;
    }

    int to_THR(double pos) const {
        return clampi((int)llround(pos), THRmin, THRmax);
    }

    double clamp_pos(double pos) const {
        if (pos < THRmin) return THRmin;
        if (pos > THRmax) return THRmax;
        return pos;
    }

    struct GWOResult {
        int best_THR;
        EvalResult best_res;
        double best_cost;
    };

    GWOResult optimize() {
        uniform_real_distribution<double> u01(0.0, 1.0);

        vector<Wolf> pack(pack_size);
        Wolf alpha, beta, delta;
        alpha.cost = beta.cost = delta.cost = numeric_limits<double>::infinity();

        int gbest_THR = THRmin;
        EvalResult gbest_res{};
        double gbest_cost = numeric_limits<double>::infinity();

        for (int i = 0; i < pack_size; i++) {
            pack[i].x = THRmin + (THRmax - THRmin) * u01(rng);
            int THR = to_THR(pack[i].x);
            EvalResult res = evaluate_burst(*studs, THR, Q,
                seed + (uint32_t)(i * 17), EPS);
            double cost = compute_cost(res);
            pack[i].cost = cost;

            if (cost < gbest_cost) {
                gbest_cost = cost;
                gbest_THR = THR;
                gbest_res = res;
            }

            if (cost < alpha.cost) {
                delta = beta; beta = alpha; alpha = pack[i];
            } else if (cost < beta.cost) {
                delta = beta; beta = pack[i];
            } else if (cost < delta.cost) {
                delta = pack[i];
            }
        }

        cerr << "[GWO] Initial best: cost=" << fixed << setprecision(6)
             << gbest_cost << " THR=" << gbest_THR << "\n";

        for (int iter = 0; iter < max_iters; iter++) {
            double a = 2.0 - 2.0 * (double)iter / max(1, max_iters - 1);

            for (int i = 0; i < pack_size; i++) {
                double Xw = pack[i].x;

                double r1 = u01(rng), r2 = u01(rng);
                double A1 = 2.0 * a * r1 - a;
                double C1 = 2.0 * r2;
                double D_alpha = fabs(C1 * alpha.x - Xw);
                double X1 = alpha.x - A1 * D_alpha;

                double r3 = u01(rng), r4 = u01(rng);
                double A2 = 2.0 * a * r3 - a;
                double C2 = 2.0 * r4;
                double D_beta = fabs(C2 * beta.x - Xw);
                double X2 = beta.x - A2 * D_beta;

                double r5 = u01(rng), r6 = u01(rng);
                double A3 = 2.0 * a * r5 - a;
                double C3 = 2.0 * r6;
                double D_delta = fabs(C3 * delta.x - Xw);
                double X3 = delta.x - A3 * D_delta;

                double Xnew = (X1 + X2 + X3) / 3.0;
                pack[i].x = clamp_pos(Xnew);

                int THR = to_THR(pack[i].x);
                EvalResult res = evaluate_burst(*studs, THR, Q,
                    seed + (uint32_t)(1000 + iter * 31 + i), EPS);
                double cost = compute_cost(res);
                pack[i].cost = cost;

                if (cost < gbest_cost) {
                    gbest_cost = cost;
                    gbest_THR = THR;
                    gbest_res = res;
                }
            }

            alpha.cost = beta.cost = delta.cost = numeric_limits<double>::infinity();
            for (int i = 0; i < pack_size; i++) {
                double cost = pack[i].cost;
                if (cost < alpha.cost) {
                    delta = beta; beta = alpha; alpha = pack[i];
                } else if (cost < beta.cost) {
                    delta = beta; beta = pack[i];
                } else if (cost < delta.cost) {
                    delta = pack[i];
                }
            }

            cerr << "[GWO] iter " << (iter + 1) << "/" << max_iters
                 << "  best_cost=" << fixed << setprecision(6) << gbest_cost
                 << "  THR=" << gbest_THR << "\n";
        }

        return GWOResult{gbest_THR, gbest_res, gbest_cost};
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
    int pack   = args.count("pack")   ? stoi(args["pack"])   : 16;
    int iters  = args.count("iters")  ? stoi(args["iters"])  : 25;

    int NUM_ITERATIONS = 2;

    cout << "============================================================\n";
    cout << "  Burst Trie + GWO Optimization (30k dataset)\n";
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
    cout << "GWO params: pack_size=" << pack << ", max_iters=" << iters
         << ", THR range=[" << tmin << "," << tmax << "]\n";
    cout << "GWO: a decreases linearly from 2 to 0 over iterations\n";
    cout << "Number of iterations: " << NUM_ITERATIONS << "\n\n";

    for (int iter = 1; iter <= NUM_ITERATIONS; iter++) {
        cout << "------------------------------------------------------------\n";
        cout << "  ITERATION " << iter << " of " << NUM_ITERATIONS << "\n";
        cout << "------------------------------------------------------------\n";

        GWOOptimizer gwo;
        gwo.THRmin = tmin;
        gwo.THRmax = tmax;
        gwo.pack_size = pack;
        gwo.max_iters = iters;
        gwo.studs = &all_students;
        gwo.Q = Q;
        gwo.seed = seed + (uint32_t)(iter * 7919);
        gwo.EPS = EPS;
        gwo.rng.seed(seed + iter * 31337u);

        auto t_opt_start = clk::now();
        auto best = gwo.optimize();
        auto t_opt_end = clk::now();
        double opt_time_ms = elapsed_ms(t_opt_start, t_opt_end);

        cout << "\n";
        cout << "  GWO Optimization complete (took " << fixed << setprecision(2)
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
        cout << "  | Burst + GWO |  Iteration " << iter << "                     |\n";
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
    cout << "  Burst Trie + GWO complete. " << NUM_ITERATIONS << " iterations done.\n";
    cout << "============================================================\n";

    return 0;
}
