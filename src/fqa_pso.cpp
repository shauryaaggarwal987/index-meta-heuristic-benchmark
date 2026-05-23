// ============================================================================
// fqa_pso.cpp — FQA (Fixed Queries Array) + PSO (Particle Swarm Optimization)
// Self-contained single file. Reads CSV directly. No students_data.hpp needed.
//
// Build:  g++ -O2 -std=gnu++17 fqa_pso.cpp -o fqa_pso
// Run:    ./fqa_pso --data=students_100k.csv --n=30000
//
// Tunes FQA pivot count K via PSO, then reports build time, query time,
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

    // === BUILD ===
    FQAIndex fqa;
    auto t0 = clk::now();
    fqa.build(students, K);
    auto t1 = clk::now();
    er.build_ms = elapsed_ms(t0, t1);

    // === MEMORY ===
    er.mem_bytes = fqa.approx_bytes();

    // === QUERY ===
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, (int)students.size() - 1);
    vector<double> cg_queries(Q);
    for (int i = 0; i < Q; i++) {
        cg_queries[i] = students[dist(rng)].cgpa;
    }

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

// ========================== PSO OPTIMIZER ====================================
// Tunes a single integer parameter K (FQA pivot count) in [Kmin, Kmax].
// Each particle has a 1-D continuous position that gets discretized to int K.

struct Particle {
    double x;           // continuous position (maps to K)
    double v;           // velocity
    double pbest_x;     // personal best position
    double pbest_cost;  // personal best cost
};

struct PSOOptimizer {
    // Search bounds
    int Kmin = 4, Kmax = 32;

    // PSO hyperparams
    int swarm_size = 16;     // number of particles
    int max_iters  = 25;     // PSO iterations

    // PSO coefficients
    double w_inertia = 0.72; // inertia weight
    double c1 = 1.49;        // cognitive coefficient
    double c2 = 1.49;        // social coefficient

    // Objective weights
    double w_build = 1.0;
    double w_query = 1.0;
    double w_mem   = 0.001;  // per MB

    // Evaluation context
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

    // Discretize continuous position to integer K
    int to_K(double pos) const {
        return clampi((int)llround(pos), Kmin, Kmax);
    }

    struct PSOResult {
        int best_K;
        EvalResult best_res;
        double best_cost;
    };

    PSOResult optimize() {
        uniform_real_distribution<double> u01(0.0, 1.0);

        // Random position in [Kmin, Kmax]
        auto rand_pos = [&]() -> double {
            return Kmin + (Kmax - Kmin) * u01(rng);
        };

        // === Initialise swarm ===
        vector<Particle> swarm(swarm_size);
        for (int i = 0; i < swarm_size; i++) {
            swarm[i].x = rand_pos();
            swarm[i].v = 0.0;
            swarm[i].pbest_x = swarm[i].x;
            swarm[i].pbest_cost = numeric_limits<double>::infinity();
        }

        double gbest_x = swarm[0].x;
        double gbest_cost = numeric_limits<double>::infinity();
        int gbest_K = to_K(gbest_x);
        EvalResult gbest_res{};

        // === Evaluate initial swarm ===
        for (int i = 0; i < swarm_size; i++) {
            int K = to_K(swarm[i].x);
            EvalResult res = evaluate_fqa(*studs, K, Q,
                seed + (uint32_t)(i * 17), EPS);
            double cost = compute_cost(res);

            swarm[i].pbest_cost = cost;
            swarm[i].pbest_x = swarm[i].x;

            if (cost < gbest_cost) {
                gbest_cost = cost;
                gbest_x = swarm[i].x;
                gbest_K = K;
                gbest_res = res;
            }
        }

        cerr << "[PSO] Initial best: cost=" << fixed << setprecision(6)
             << gbest_cost << " K=" << gbest_K << "\n";

        // === Main PSO loop ===
        for (int it = 1; it <= max_iters; it++) {
            for (int i = 0; i < swarm_size; i++) {
                // Update velocity
                double r1 = u01(rng);
                double r2 = u01(rng);
                swarm[i].v = w_inertia * swarm[i].v
                           + c1 * r1 * (swarm[i].pbest_x - swarm[i].x)
                           + c2 * r2 * (gbest_x - swarm[i].x);

                // Update position
                swarm[i].x += swarm[i].v;

                // Clamp to bounds
                if (swarm[i].x < Kmin) swarm[i].x = Kmin;
                if (swarm[i].x > Kmax) swarm[i].x = Kmax;

                // Evaluate
                int K = to_K(swarm[i].x);
                EvalResult res = evaluate_fqa(*studs, K, Q,
                    seed + (uint32_t)(12345 + it * 257 + i * 17), EPS);
                double cost = compute_cost(res);

                // Update personal best
                if (cost < swarm[i].pbest_cost) {
                    swarm[i].pbest_cost = cost;
                    swarm[i].pbest_x = swarm[i].x;
                }

                // Update global best
                if (cost < gbest_cost) {
                    gbest_cost = cost;
                    gbest_x = swarm[i].x;
                    gbest_K = K;
                    gbest_res = res;
                }
            }

            cerr << "[PSO] iter " << it << "/" << max_iters
                 << "  best_cost=" << fixed << setprecision(6) << gbest_cost
                 << "  K=" << gbest_K << "\n";
        }

        return PSOResult{gbest_K, gbest_res, gbest_cost};
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

    // PSO hyperparams (overridable)
    int kmin   = args.count("kmin")   ? stoi(args["kmin"])   : 4;
    int kmax   = args.count("kmax")   ? stoi(args["kmax"])   : 32;
    int swarm  = args.count("swarm")  ? stoi(args["swarm"])  : 16;
    int iters  = args.count("iters")  ? stoi(args["iters"])  : 25;

    int NUM_ITERATIONS = 2;  // run the full process twice for genuine data

    // === Load dataset ===
    cout << "============================================================\n";
    cout << "  FQA + PSO Optimization (30k dataset)\n";
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
    cout << "PSO params: swarm_size=" << swarm << ", max_iters=" << iters
         << ", K range=[" << kmin << "," << kmax << "]\n";
    cout << "PSO coefficients: w_inertia=0.72, c1=1.49, c2=1.49\n";
    cout << "Number of iterations: " << NUM_ITERATIONS << "\n\n";

    // === Run NUM_ITERATIONS iterations ===
    for (int iter = 1; iter <= NUM_ITERATIONS; iter++) {
        cout << "------------------------------------------------------------\n";
        cout << "  ITERATION " << iter << " of " << NUM_ITERATIONS << "\n";
        cout << "------------------------------------------------------------\n";

        PSOOptimizer pso;
        pso.Kmin = kmin;
        pso.Kmax = kmax;
        pso.swarm_size = swarm;
        pso.max_iters = iters;
        pso.studs = &all_students;
        pso.Q = Q;
        pso.seed = seed + (uint32_t)(iter * 7919);
        pso.EPS = EPS;
        pso.rng.seed(seed + iter * 31337u);

        // Run PSO optimization
        auto t_opt_start = clk::now();
        auto best = pso.optimize();
        auto t_opt_end = clk::now();
        double opt_time_ms = elapsed_ms(t_opt_start, t_opt_end);

        cout << "\n";
        cout << "  PSO Optimization complete (took " << fixed << setprecision(2)
             << opt_time_ms << " ms)\n";
        cout << "  Best K found: " << best.best_K << "\n\n";

        // === Final evaluation with the best K (fresh, independent measurement) ===
        cout << "  --- Final Evaluation with optimized K=" << best.best_K << " ---\n";

        // Build timing
        FQAIndex fqa;
        auto tb0 = clk::now();
        fqa.build(all_students, best.best_K);
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
        cout << "  | FQA + PSO  |  Iteration " << iter << "                      |\n";
        cout << "  +-------------------------------------------------+\n";
        cout << "  | Optimized K (pivots)  : " << best.best_K << "\n";
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
    cout << "  FQA + PSO complete. " << NUM_ITERATIONS << " iterations done.\n";
    cout << "============================================================\n";

    return 0;
}
