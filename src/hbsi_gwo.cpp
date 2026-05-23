// ============================================================================
// hbsi_gwo.cpp — Hash-Bucketed String Index (HBSI) + GWO
// Self-contained single file. Reads CSV directly. No students_data.hpp needed.
//
// Build:  g++ -O2 -std=gnu++17 hbsi_gwo.cpp -o hbsi_gwo
// Run:    ./hbsi_gwo --data=students_100k.csv --n=30000
//
// Tunes the HBSI bucket size C via Grey Wolf Optimizer (GWO), then reports
// build time, query time, and estimated index memory for two trials by default.
// HBSI is a hash-bucketed exact-match string index over registration numbers;
// it is not an implementation of Willard's original HBSI.
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

// ========================== HBSI (on hashed RegNo) ===================
struct HBSI {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    static uint64_t key_from_reg(const string &s) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (int i = 0; i < (int)s.size(); i++) {
            unsigned char c = (unsigned char)s[i];
            h ^= c;
            h *= 0x100000001b3ULL;
        }
        return splitmix64(h);
    }

    struct Rep { uint64_t key; int bucket_id; };
    struct Bucket { vector<pair<uint64_t, int>> items; };

    const vector<Student> *students = nullptr;
    vector<uint64_t> keys;
    vector<int> order_by_key;
    vector<Rep> reps;
    vector<Bucket> buckets;
    int c = 64;

    void build(const vector<Student> &st, int c_bucket = 64) {
        students = &st;
        c = max(1, c_bucket);

        int n = (int)st.size();
        keys.resize(n);
        for (int i = 0; i < n; i++) keys[i] = key_from_reg(st[i].reg_no);

        order_by_key.resize(n);
        for (int i = 0; i < n; i++) order_by_key[i] = i;

        sort(order_by_key.begin(), order_by_key.end(), [&](int a, int b) {
            if (keys[a] != keys[b]) return keys[a] < keys[b];
            return st[a].reg_no < st[b].reg_no;
        });

        reps.clear();
        buckets.clear();

        for (int i = 0; i < n; i += c) {
            int first = i;
            int last = min(n, i + c) - 1;

            int rep_id = order_by_key[first];
            uint64_t krep = keys[rep_id];

            int bucket_id = (int)buckets.size();
            reps.push_back(Rep{krep, bucket_id});
            buckets.push_back(Bucket{});

            auto &B = buckets.back().items;
            B.reserve(last - first + 1);

            for (int t = first; t <= last; t++) {
                int sid = order_by_key[t];
                B.push_back({keys[sid], sid});
            }
        }

        if (reps.empty() && n > 0) {
            reps.push_back(Rep{keys[order_by_key[0]], 0});
            buckets.push_back(Bucket{});
        }
    }

    int find_exact(const string &reg_no) const {
        if (!students || students->empty() || reps.empty() || buckets.empty()) return -1;

        uint64_t kq = key_from_reg(reg_no);

        int R = (int)(upper_bound(reps.begin(), reps.end(), kq,
                    [](uint64_t val, const Rep &r) { return val < r.key; }) - reps.begin()) - 1;

        if (R < 0) R = 0;
        if (R >= (int)reps.size()) R = (int)reps.size() - 1;

        auto search_bucket = [&](int rep_index) -> int {
            if (rep_index < 0 || rep_index >= (int)reps.size()) return -1;

            const auto &B = buckets[reps[rep_index].bucket_id].items;
            if (B.empty()) return -1;

            auto cmp = [](const pair<uint64_t, int> &a, const pair<uint64_t, int> &b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            };

            auto it_lb = lower_bound(B.begin(), B.end(), make_pair(kq, -1), cmp);

            for (auto jt = it_lb; jt != B.end() && jt->first == kq; ++jt) {
                int sid = jt->second;
                if ((*students)[sid].reg_no == reg_no) return sid;
            }

            return -1;
        };

        // Normal case: the selected bucket contains the key range.
        int ans = search_bucket(R);
        if (ans != -1) return ans;

        // Robustness for the rare case of equal hash keys crossing bucket boundaries.
        // With 64-bit hashes this is extremely unlikely for the present dataset, but
        // checking adjacent equal-key boundary buckets prevents false negatives.
        for (int b = R - 1; b >= 0; --b) {
            const auto &B = buckets[reps[b].bucket_id].items;
            if (B.empty() || B.back().first < kq) break;
            ans = search_bucket(b);
            if (ans != -1) return ans;
            if (B.front().first < kq) break;
        }

        for (int b = R + 1; b < (int)reps.size(); ++b) {
            const auto &B = buckets[reps[b].bucket_id].items;
            if (B.empty() || B.front().first > kq) break;
            ans = search_bucket(b);
            if (ans != -1) return ans;
            if (B.back().first > kq) break;
        }

        return -1;
    }

    size_t approx_bytes() const {
        size_t bytes = 0;
        bytes += keys.capacity() * sizeof(uint64_t);
        bytes += order_by_key.capacity() * sizeof(int);
        bytes += reps.capacity() * sizeof(Rep);
        for (int i = 0; i < (int)buckets.size(); i++) {
            bytes += buckets[i].items.capacity() * sizeof(pair<uint64_t, int>);
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

static EvalResult evaluate_hbsi(const vector<Student> &students, int Csize,
                                int Q, uint32_t seed)
{
    EvalResult er{};
    static volatile size_t sink = 0;

    HBSI hbsi;
    auto t0 = clk::now();
    hbsi.build(students, Csize);
    auto t1 = clk::now();
    er.build_ms = elapsed_ms(t0, t1);

    er.mem_bytes = hbsi.approx_bytes();

    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, (int)students.size() - 1);
    vector<string> reg_queries(Q);
    for (int i = 0; i < Q; i++) {
        reg_queries[i] = students[dist(rng)].reg_no;
    }

    auto q0 = clk::now();
    for (int i = 0; i < Q; i++) {
        volatile int idx = hbsi.find_exact(reg_queries[i]);
        sink += (size_t)(idx != -1);
    }
    auto q1 = clk::now();
    er.query_ms = elapsed_ms(q0, q1);

    do_not_optimize_away(&sink);
    return er;
}

// ========================== GWO OPTIMIZER ====================================
struct Wolf {
    double x = 0.0;
    double cost = numeric_limits<double>::infinity();
};

struct GWOOptimizer {
    int Cmin = 16, Cmax = 256;
    int pack_size = 16;
    int max_iters = 25;

    double w_build = 1.0;
    double w_query = 1.0;
    double w_mem   = 0.001;

    const vector<Student> *studs = nullptr;
    int Q = 2000;
    uint32_t seed = 42u;
    mt19937 rng{123};

    static int clampi(int v, int lo, int hi) { return max(lo, min(hi, v)); }

    double compute_cost(const EvalResult &r) const {
        double mem_mb = (double)r.mem_bytes / (1024.0 * 1024.0);
        return w_build * r.build_ms + w_query * r.query_ms + w_mem * mem_mb;
    }

    int to_C(double pos) const {
        return clampi((int)llround(pos), Cmin, Cmax);
    }

    double clamp_pos(double pos) const {
        if (pos < Cmin) return Cmin;
        if (pos > Cmax) return Cmax;
        return pos;
    }

    struct GWOResult {
        int best_C;
        EvalResult best_res;
        double best_cost;
    };

    GWOResult optimize() {
        uniform_real_distribution<double> u01(0.0, 1.0);

        vector<Wolf> pack(pack_size);
        Wolf alpha, beta, delta;
        alpha.cost = beta.cost = delta.cost = numeric_limits<double>::infinity();

        int gbest_C = Cmin;
        EvalResult gbest_res{};
        double gbest_cost = numeric_limits<double>::infinity();

        for (int i = 0; i < pack_size; i++) {
            pack[i].x = Cmin + (Cmax - Cmin) * u01(rng);
            int C = to_C(pack[i].x);
            EvalResult res = evaluate_hbsi(*studs, C, Q,
                seed + (uint32_t)(i * 17));
            double cost = compute_cost(res);
            pack[i].cost = cost;

            if (cost < gbest_cost) {
                gbest_cost = cost;
                gbest_C = C;
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
             << gbest_cost << " C=" << gbest_C << "\n";

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

                int C = to_C(pack[i].x);
                EvalResult res = evaluate_hbsi(*studs, C, Q,
                    seed + (uint32_t)(1000 + iter * 31 + i));
                double cost = compute_cost(res);
                pack[i].cost = cost;

                if (cost < gbest_cost) {
                    gbest_cost = cost;
                    gbest_C = C;
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
                 << "  C=" << gbest_C << "\n";
        }

        return GWOResult{gbest_C, gbest_res, gbest_cost};
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
    int cmin   = args.count("cmin")   ? stoi(args["cmin"])   : 16;
    int cmax   = args.count("cmax")   ? stoi(args["cmax"])   : 256;
    int pack   = args.count("pack")   ? stoi(args["pack"])   : 16;
    int iters  = args.count("iters")  ? stoi(args["iters"])  : 25;

    int NUM_TRIALS = args.count("trials") ? stoi(args["trials"]) : 2;
    if (NUM_TRIALS <= 0) { cerr << "ERROR: --trials must be positive.\n"; return 1; }
    if (N <= 0) { cerr << "ERROR: --n must be positive.\n"; return 1; }
    if (Q <= 0) { cerr << "ERROR: --Q must be positive.\n"; return 1; }
    if (cmin <= 0 || cmax < cmin) { cerr << "ERROR: invalid C range.\n"; return 1; }
    if (pack < 3) { cerr << "ERROR: --pack must be at least 3 for GWO alpha/beta/delta wolves.\n"; return 1; }
    if (iters <= 0) { cerr << "ERROR: --iters must be positive.\n"; return 1; }

    cout << "============================================================\n";
    cout << "  HBSI + GWO Optimization\n";
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
         << ", C range=[" << cmin << "," << cmax << "]\n";
    cout << "GWO: a decreases linearly from 2 to 0 over iterations\n";
    cout << "Number of trials: " << NUM_TRIALS << "\n\n";

    for (int iter = 1; iter <= NUM_TRIALS; iter++) {
        cout << "------------------------------------------------------------\n";
        cout << "  TRIAL " << iter << " of " << NUM_TRIALS << "\n";
        cout << "------------------------------------------------------------\n";

        GWOOptimizer gwo;
        gwo.Cmin = cmin;
        gwo.Cmax = cmax;
        gwo.pack_size = pack;
        gwo.max_iters = iters;
        gwo.studs = &all_students;
        gwo.Q = Q;
        gwo.seed = seed + (uint32_t)(iter * 7919);
        gwo.rng.seed(seed + iter * 31337u);

        auto t_opt_start = clk::now();
        auto best = gwo.optimize();
        auto t_opt_end = clk::now();
        double opt_time_ms = elapsed_ms(t_opt_start, t_opt_end);

        cout << "\n";
        cout << "  GWO Optimization complete (took " << fixed << setprecision(2)
             << opt_time_ms << " ms)\n";
        cout << "  Best C (bucket size) found: " << best.best_C << "\n\n";

        cout << "  --- Final Evaluation with optimized C=" << best.best_C << " ---\n";

        HBSI hbsi;
        auto tb0 = clk::now();
        hbsi.build(all_students, best.best_C);
        auto tb1 = clk::now();
        double final_build_ms = elapsed_ms(tb0, tb1);

        size_t final_mem = hbsi.approx_bytes();

        mt19937 qrng(seed + iter * 9973u);
        uniform_int_distribution<int> qdist(0, (int)all_students.size() - 1);
        vector<string> reg_queries(Q);
        for (int i = 0; i < Q; i++) {
            reg_queries[i] = all_students[qdist(qrng)].reg_no;
        }

        volatile size_t sink = 0;
        auto tq0 = clk::now();
        for (int i = 0; i < Q; i++) {
            volatile int idx = hbsi.find_exact(reg_queries[i]);
            sink += (size_t)(idx != -1);
        }
        auto tq1 = clk::now();
        double final_query_ms = elapsed_ms(tq0, tq1);
        do_not_optimize_away(&sink);

        cout << "\n";
        cout << "  +-------------------------------------------------+\n";
        cout << "  | HBSI + GWO |  Trial " << iter << "                     |\n";
        cout << "  +-------------------------------------------------+\n";
        cout << "  | Optimized C (bucket)  : " << best.best_C << "\n";
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
    cout << "  HBSI + GWO complete. " << NUM_TRIALS << " trials done.\n";
    cout << "============================================================\n";

    return 0;
}
