// ============================================================================
// fqa_abc.cpp — FQA (Fixed Queries Array) + ABC (Artificial Bee Colony)
// Self-contained single file. Reads CSV directly. No students_data.hpp needed.
//
// Build:  g++ -O2 -std=gnu++17 fqa_abc.cpp -o fqa_abc
// Run:    ./fqa_abc --data=students_100k.csv --n=30000
//
// Tunes FQA pivot count K via ABC, then reports build time, query time,
// and memory for 2 full iterations.
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
        // strip \r if present (Windows line endings)
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto cols = split_csv_line(line);
        if ((int)cols.size() < 3) continue;
        if (!header_skipped) {
            // Check if first row is a header
            try { (void)stod(cols[2]); }
            catch (...) { header_skipped = true; continue; }
            header_skipped = true;
            // If stod succeeded, this row is data — fall through
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

// ========================== FQA INDEX (over CGPA) ===========================
struct FQAIndex {
    const vector<Student> *students = nullptr;
    vector<int> pivot_idx;
    vector<int> order;
    vector< array<float, 32> > coords_small;
    int k = 0;

    static vector<int> choose_pivots(const vector<Student> &st, int k) {
        int n = (int)st.size();
        vector<int> piv;
        if (n == 0 || k <= 0) return piv;

        vector<int> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i;

        sort(idx.begin(), idx.end(), [&](int a, int b) {
            return st[a].cgpa < st[b].cgpa;
        });

        piv.push_back(idx[n / 2]);

        auto dist = [&](int i, int p) {
            return fabs(st[i].cgpa - st[p].cgpa);
        };

        while ((int)piv.size() < k) {
            int best = -1;
            double bestd = -1;
            for (int i = 0; i < n; i++) {
                double dmin = 1e100;
                for (int j = 0; j < (int)piv.size(); j++) {
                    dmin = min(dmin, dist(i, piv[j]));
                }
                if (dmin > bestd) { bestd = dmin; best = i; }
            }
            if (best == -1) break;
            piv.push_back(best);
        }
        return piv;
    }

    void build(const vector<Student> &st, int k_pivots = 16) {
        students = &st;
        int n = (int)st.size();
        k = max(1, min(32, min(k_pivots, max(1, n))));
        pivot_idx = choose_pivots(st, k);

        order.resize(n);
        for (int i = 0; i < n; i++) order[i] = i;

        coords_small.assign(n, array<float, 32>{});
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                coords_small[i][j] = (float)fabs(st[i].cgpa - st[pivot_idx[j]].cgpa);
            }
        }

        sort(order.begin(), order.end(), [&](int a, int b) {
            for (int j = 0; j < k; j++) {
                if (coords_small[a][j] < coords_small[b][j]) return true;
                if (coords_small[a][j] > coords_small[b][j]) return false;
            }
            if (st[a].reg_no != st[b].reg_no) return st[a].reg_no < st[b].reg_no;
            return st[a].name < st[b].name;
        });
    }

    array<float, 32> target_coords(double cg) const {
        array<float, 32> t{};
        for (int j = 0; j < k; j++)
            t[j] = (float)fabs(cg - students->at(pivot_idx[j]).cgpa);
        return t;
    }

    bool coords_lt(int idx, const array<float, 32> &T) const {
        const auto &A = coords_small[idx];
        for (int j = 0; j < k; j++) {
            if (A[j] < T[j]) return true;
            if (A[j] > T[j]) return false;
        }
        return false;
    }

    vector<int> query_exact(double cg, double eps = 1e-9) const {
        if (!students) return {};
        int n = (int)order.size();
        if (n == 0) return {};

        auto T = target_coords(cg);

        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            int idx = order[mid];
            if (coords_lt(idx, T)) lo = mid + 1;
            else hi = mid;
        }

        vector<int> ans;
        for (int dir = -1; dir <= 1; dir += 2) {
            int i = lo + (dir == 1 ? 0 : -1);
            while (i >= 0 && i < n) {
                int idx = order[i];
                double g = students->at(idx).cgpa;
                if (fabs(g - cg) <= eps) ans.push_back(idx);
                else {
                    float a0 = coords_small[idx][0], t0 = T[0];
                    if (fabs(a0 - t0) > 1e-4) break;
                }
                i += dir;
            }
        }
        return ans;
    }

    size_t approx_bytes() const {
        size_t bytes = 0;
        bytes += pivot_idx.capacity() * sizeof(int);
        bytes += order.capacity() * sizeof(int);
        bytes += coords_small.capacity() * sizeof(array<float, 32>);
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
// Evaluates a given K value on the FQA index: returns {build_ms, query_ms, mem_bytes}
struct EvalResult {
    double build_ms;
    double query_ms;    // total time for Q queries
    size_t mem_bytes;
};

static EvalResult evaluate_fqa(const vector<Student> &students, int K,
                               int Q, uint32_t seed, double EPS)
{
    EvalResult er{};
    static volatile size_t sink = 0;

    // === BUILD: time the build ===
    FQAIndex fqa;
    auto t0 = clk::now();
    fqa.build(students, K);
    auto t1 = clk::now();
    er.build_ms = elapsed_ms(t0, t1);

    // === MEMORY: resident size of the index ===
    er.mem_bytes = fqa.approx_bytes();

    // === QUERY: prepare random queries sampled from dataset ===
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, (int)students.size() - 1);
    vector<double> cg_queries(Q);
    for (int i = 0; i < Q; i++) {
        cg_queries[i] = students[dist(rng)].cgpa;
    }

    // Time Q exact queries
    auto q0 = clk::now();
    for (int i = 0; i < Q; i++) {
        auto res = fqa.query_exact(cg_queries[i], EPS);
        sink += res.size();
    }
    auto q1 = clk::now();
    er.query_ms = elapsed_ms(q0, q1);

    do_not_optimize_away(&sink);
    return er;
}

// ========================== ABC OPTIMIZER ====================================
// Tunes a single integer parameter K (FQA pivot count) in [Kmin, Kmax].
// Objective: weighted combination of build time, query time, and memory.

struct ABCFood {
    int K;
    EvalResult res;
    double cost;
    int trials;
};

struct ABCOptimizer {
    // Search bounds
    int Kmin = 4, Kmax = 32;

    // ABC hyperparams
    int SN = 12;             // number of food sources
    int limit = 8;           // scout abandon threshold
    int max_cycles = 25;     // ABC iterations

    // Objective weights
    double w_build = 1.0;
    double w_query = 1.0;
    double w_mem   = 0.001;  // per MB

    // Evaluation context
    const vector<Student> *studs = nullptr;
    int Q = 2000;
    uint32_t seed = 42u;
    double EPS = 1e-9;

    mt19937 rng{1234567};

    int rnd_int(int a, int b) {
        uniform_int_distribution<int> d(a, b);
        return d(rng);
    }
    double rnd01() {
        uniform_real_distribution<double> d(0.0, 1.0);
        return d(rng);
    }
    double rnd_sym() { return rnd01() * 2.0 - 1.0; }

    static int clampi(int v, int lo, int hi) { return max(lo, min(hi, v)); }

    double compute_cost(const EvalResult &r) const {
        double mem_mb = (double)r.mem_bytes / (1024.0 * 1024.0);
        return w_build * r.build_ms + w_query * r.query_ms + w_mem * mem_mb;
    }

    ABCFood make_food(int K) {
        ABCFood f;
        f.K = K;
        f.res = evaluate_fqa(*studs, K, Q, seed, EPS);
        f.cost = compute_cost(f.res);
        f.trials = 0;
        return f;
    }

    ABCFood random_food() {
        return make_food(rnd_int(Kmin, Kmax));
    }

    ABCFood mutate_neighbor(const ABCFood &base, const ABCFood &other) {
        double phi = rnd_sym();
        double v = base.K + phi * (base.K - other.K);
        int newK = clampi((int)llround(v), Kmin, Kmax);
        return make_food(newK);
    }

    ABCFood optimize() {
        // Initialise food sources
        vector<ABCFood> foods;
        foods.reserve(SN);
        for (int i = 0; i < SN; i++) {
            foods.push_back(random_food());
        }

        ABCFood gbest = *min_element(foods.begin(), foods.end(),
            [](const ABCFood &a, const ABCFood &b) { return a.cost < b.cost; });

        for (int cy = 0; cy < max_cycles; cy++) {
            // === Employed bees phase ===
            for (int i = 0; i < SN; i++) {
                int j = i;
                while (j == i) j = rnd_int(0, SN - 1);
                ABCFood cand = mutate_neighbor(foods[i], foods[j]);
                if (cand.cost < foods[i].cost) {
                    foods[i] = cand;
                    foods[i].trials = 0;
                    if (cand.cost < gbest.cost) gbest = cand;
                } else {
                    foods[i].trials++;
                }
            }

            // === Onlooker bees phase ===
            vector<double> q(SN);
            double sumq = 0.0;
            for (int i = 0; i < SN; i++) {
                q[i] = 1.0 / (1.0 + foods[i].cost);
                sumq += q[i];
            }
            for (int r = 0; r < SN; r++) {
                double pick = rnd01() * sumq, acc = 0.0;
                int i = 0;
                for (; i < SN; i++) { acc += q[i]; if (acc >= pick) break; }
                if (i >= SN) i = SN - 1;

                int j = i;
                while (j == i) j = rnd_int(0, SN - 1);
                ABCFood cand = mutate_neighbor(foods[i], foods[j]);
                if (cand.cost < foods[i].cost) {
                    foods[i] = cand;
                    foods[i].trials = 0;
                    if (cand.cost < gbest.cost) gbest = cand;
                } else {
                    foods[i].trials++;
                }
            }

            // === Scout bees phase ===
            for (int i = 0; i < SN; i++) {
                if (foods[i].trials > limit) {
                    foods[i] = random_food();
                    if (foods[i].cost < gbest.cost) gbest = foods[i];
                }
            }

            cerr << "[ABC] cycle " << (cy + 1) << "/" << max_cycles
                 << "  best_cost=" << fixed << setprecision(6) << gbest.cost
                 << "  K=" << gbest.K << "\n";
        }

        return gbest;
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

    // Dataset path
    string data_path = args.count("data") ? args["data"] : "students_100k.csv";
    int N = args.count("n") ? stoi(args["n"]) : 30000;
    uint32_t seed = args.count("seed") ? (uint32_t)stoul(args["seed"]) : 42u;
    int Q = args.count("Q") ? stoi(args["Q"]) : 2000;
    double EPS = args.count("eps") ? stod(args["eps"]) : 1e-9;

    // ABC hyperparams (overridable)
    int kmin   = args.count("kmin")   ? stoi(args["kmin"])   : 4;
    int kmax   = args.count("kmax")   ? stoi(args["kmax"])   : 32;
    int sn     = args.count("sn")     ? stoi(args["sn"])     : 12;
    int limit  = args.count("limit")  ? stoi(args["limit"])  : 8;
    int cycles = args.count("cycles") ? stoi(args["cycles"]) : 25;

    int NUM_ITERATIONS = 2;  // run the full process twice for genuine data

    // === Load dataset ===
    cout << "============================================================\n";
    cout << "  FQA + ABC Optimization (30k dataset)\n";
    cout << "============================================================\n\n";

    vector<Student> all_students = load_students_csv(data_path);
    if (all_students.empty()) {
        cerr << "ERROR: Could not load dataset from: " << data_path << "\n";
        return 1;
    }
    cout << "Loaded " << all_students.size() << " students from: " << data_path << "\n";

    // Take N records (deterministic shuffle)
    if (N > 0 && N < (int)all_students.size()) {
        mt19937 rng_shuffle(seed);
        shuffle(all_students.begin(), all_students.end(), rng_shuffle);
        all_students.resize(N);
    }
    cout << "Using subset: N = " << all_students.size() << "\n";
    cout << "Queries per evaluation: Q = " << Q << "\n";
    cout << "ABC params: SN=" << sn << ", limit=" << limit
         << ", cycles=" << cycles << ", K range=[" << kmin << "," << kmax << "]\n";
    cout << "Number of iterations: " << NUM_ITERATIONS << "\n\n";

    // === Run NUM_ITERATIONS iterations ===
    for (int iter = 1; iter <= NUM_ITERATIONS; iter++) {
        cout << "------------------------------------------------------------\n";
        cout << "  ITERATION " << iter << " of " << NUM_ITERATIONS << "\n";
        cout << "------------------------------------------------------------\n";

        // Vary the ABC's internal RNG seed per iteration for different exploration
        ABCOptimizer abc;
        abc.Kmin = kmin;
        abc.Kmax = kmax;
        abc.SN = sn;
        abc.limit = limit;
        abc.max_cycles = cycles;
        abc.studs = &all_students;
        abc.Q = Q;
        abc.seed = seed + (uint32_t)(iter * 7919);  // different seed per iteration
        abc.EPS = EPS;
        abc.rng.seed(seed + iter * 31337u);          // different ABC RNG per iteration

        // Run ABC optimization
        auto t_opt_start = clk::now();
        ABCFood best = abc.optimize();
        auto t_opt_end = clk::now();
        double opt_time_ms = elapsed_ms(t_opt_start, t_opt_end);

        cout << "\n";
        cout << "  ABC Optimization complete (took " << fixed << setprecision(2)
             << opt_time_ms << " ms)\n";
        cout << "  Best K found: " << best.K << "\n\n";

        // === Final evaluation with the best K (fresh, independent measurement) ===
        // We do a clean build + query measurement with the optimized K
        cout << "  --- Final Evaluation with optimized K=" << best.K << " ---\n";

        // Build timing
        FQAIndex fqa;
        auto tb0 = clk::now();
        fqa.build(all_students, best.K);
        auto tb1 = clk::now();
        double final_build_ms = elapsed_ms(tb0, tb1);

        // Memory
        size_t final_mem = fqa.approx_bytes();

        // Query timing
        mt19937 qrng(seed + iter * 9973u);
        uniform_int_distribution<int> qdist(0, (int)all_students.size() - 1);
        vector<double> cg_queries(Q);
        for (int i = 0; i < Q; i++) {
            cg_queries[i] = all_students[qdist(qrng)].cgpa;
        }

        volatile size_t sink = 0;
        auto tq0 = clk::now();
        for (int i = 0; i < Q; i++) {
            auto res = fqa.query_exact(cg_queries[i], EPS);
            sink += res.size();
        }
        auto tq1 = clk::now();
        double final_query_ms = elapsed_ms(tq0, tq1);
        do_not_optimize_away(&sink);

        // === Print results ===
        cout << "\n";
        cout << "  +-------------------------------------------------+\n";
        cout << "  | FQA + ABC  |  Iteration " << iter << "                      |\n";
        cout << "  +-------------------------------------------------+\n";
        cout << "  | Optimized K (pivots)  : " << best.K << "\n";
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
    cout << "  FQA + ABC complete. " << NUM_ITERATIONS << " iterations done.\n";
    cout << "============================================================\n";

    return 0;
}
