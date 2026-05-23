#include <bits/stdc++.h>
using namespace std;

/*
  Benchmarking + Query Timing + Memory Estimation for three indices over a student dataset.
  Payload per record (STRICTLY 3 fields): {name, reg_no, cgpa}

  NEW (for Step-1):
  - --n=10000          -> run benchmark on only N records (after deterministic shuffle)
  - --csv=results/x.csv-> append 3 rows (FQA/Burst/QFast) to a CSV file
  - --opt=Baseline/ABC/PSO/GWO -> label written into CSV

  Example:
    g++ -O2 -std=gnu++17 benchmarking_code.cpp -o bench
    ./bench --n=10000 --opt=Baseline --csv=results/dataset_10k.csv
*/

struct Student {
    string name;
    string reg_no;
    double cgpa;
};

// Your generated header (must exist in your project)
#include "students_data.hpp" // inline vector<Student> load_students_static()

// --------------------- Misc utils ---------------------
static inline string trim(const string &s) {
    size_t i = 0, j = s.size();
    while (i < j && isspace((unsigned char)s[i])) ++i;
    while (j > i && isspace((unsigned char)s[j - 1])) --j;
    return s.substr(i, j - i);
}

static vector<string> split_csv_line(const string &line) {
    vector<string> out;
    string cur;
    bool inq = false;
    for (int i = 0; i < (int)line.size(); i++) {
        char c = line[i];
        if (c == '\"') inq = !inq;
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
    if (!fin.is_open()) return v;

    string line;
    bool header_unknown = true;

    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto cols = split_csv_line(line);
        if ((int)cols.size() < 3) continue;

        if (header_unknown) {
            try { (void)stod(cols[2]); }
            catch (...) { header_unknown = false; continue; }
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
    static const char *S[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0;
    double d = (double)x;
    while (d >= 1024.0 && i < 4) {
        d /= 1024.0;
        i++;
    }
    ostringstream os;
    os.setf(std::ios::fixed);
    os << setprecision(d >= 100 ? 0 : (d >= 10 ? 1 : 2)) << d << " " << S[i];
    return os.str();
}

static bool file_nonempty(const string &path) {
    ifstream fin(path);
    return fin.good() && fin.peek() != EOF;
}

static void append_row_csv(const string &csv_path,
                           const string &structure,
                           const string &optimizer,
                           double build_ms,
                           double query_ms,
                           size_t mem_bytes) {
    bool need_header = !file_nonempty(csv_path);
    ofstream fout(csv_path, ios::app);
    if (!fout.is_open()) {
        cerr << "ERROR: Could not open CSV file: " << csv_path
             << "\nCreate the folder (e.g., results/) first.\n";
        return;
    }

    if (need_header) {
        fout << "Structure,Optimizer,Build(ms),Query(ms),Memory(MB)\n";
    }

    double mem_mb = (double)mem_bytes / (1024.0 * 1024.0);
    fout << structure << ","
         << optimizer << ","
         << fixed << setprecision(6) << build_ms << ","
         << fixed << setprecision(6) << query_ms << ","
         << fixed << setprecision(6) << mem_mb << "\n";
}

// --------------------- 1) FQA over CGPA ---------------------
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
                if (dmin > bestd) {
                    bestd = dmin;
                    best = i;
                }
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
        for (int j = 0; j < k; j++) {
            t[j] = (float)fabs(cg - students->at(pivot_idx[j]).cgpa);
        }
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
        bytes += coords_small.capacity() * sizeof(coords_small[0]);
        return bytes;
    }
};

// --------------------- 2) Burst Trie over Name ---------------------
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
    }

    void _burst(int u, const vector<int> &ids, size_t depth) {
        nodes[u].isLeaf = false;
        nodes[u].nxt.clear();

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

// --------------------- 3) Q-Fast-like Trie on hashed RegNo ---------------------
struct QFastTrie {
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
        c = max(8, c_bucket);

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

        const auto &B = buckets[reps[R].bucket_id].items;

        auto cmp = [](const pair<uint64_t, int> &a, const pair<uint64_t, int> &b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        };

        auto it_lb = lower_bound(B.begin(), B.end(), make_pair(kq, -1), cmp);

        for (auto jt = it_lb; jt != B.end() && jt->first == kq; ++jt) {
            int sid = jt->second;
            if (students->at(sid).reg_no == reg_no) return sid;
        }

        if (it_lb != B.begin()) {
            auto jt = it_lb;
            --jt;
            while (true) {
                if (jt->first != kq) break;
                int sid = jt->second;
                if (students->at(sid).reg_no == reg_no) return sid;
                if (jt == B.begin()) break;
                --jt;
            }
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

// --------------------- Benchmark harness ---------------------
struct Timer {
    chrono::high_resolution_clock::time_point t0;
    void tic() { t0 = chrono::high_resolution_clock::now(); }
    double toc_ms() const {
        auto t1 = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(t1 - t0).count();
    }
};

static map<string, string> parse_args(int argc, char **argv) {
    map<string, string> m;
    for (int i = 1; i < argc; i++) {
        string s(argv[i]);
        if (s.rfind("--", 0) == 0) {
            auto eq = s.find('=');
            if (eq == string::npos) m[s.substr(2)] = "1";
            else m[s.substr(2, eq - 2)] = s.substr(eq + 1);
        }
    }
    return m;
}

// --------------------- main ---------------------
int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);

    int K = args.count("k") ? stoi(args["k"]) : 16;
    int THR = args.count("thr") ? stoi(args["thr"]) : 32;
    int Csize = args.count("c") ? stoi(args["c"]) : 64;
    int Q = args.count("Q") ? stoi(args["Q"]) : 2000;
    uint32_t seed = args.count("seed") ? (uint32_t)stoul(args["seed"]) : 42u;
    double EPS = args.count("eps") ? stod(args["eps"]) : 1e-9;

    int N = args.count("n") ? stoi(args["n"]) : -1;         // dataset size
    string csv = args.count("csv") ? args["csv"] : "";       // output csv
    string opt = args.count("opt") ? args["opt"] : "Baseline";

    vector<Student> students;

    // Optional: allow external CSV data (but static dataset remains default)
    if (args.count("data")) {
        students = load_students_csv(args["data"]);
        if (students.empty()) {
            cerr << "ERROR: Could not load from --data=" << args["data"] << "\n";
            return 1;
        }
        cout << "Loaded students from CSV: " << students.size() << "\n";
    } else {
        students = load_students_static();
        if (students.empty()) {
            cerr << "ERROR: Static dataset is empty. Ensure students_data.hpp exists.\n";
            return 1;
        }
        cout << "Loaded students (static): " << students.size() << "\n";
    }

    // Deterministic subset (for 10k/20k/30k)
    if (N > 0 && N < (int)students.size()) {
        mt19937 rng(seed);
        shuffle(students.begin(), students.end(), rng);
        students.resize(N);
        cout << "Using subset N=" << N << "\n";
    } else {
        cout << "Using full dataset N=" << students.size() << "\n";
    }

    // ---------- Build indices ----------
    Timer t;

    // FQA
    FQAIndex fqa;
    t.tic();
    fqa.build(students, K);
    double fqa_build_ms = t.toc_ms();
    size_t fqa_mem = fqa.approx_bytes();

    // Burst
    BurstTrie btrie(THR);
    t.tic();
    btrie.build(students, THR);
    double bt_build_ms = t.toc_ms();
    size_t bt_mem = btrie.approx_bytes();

    // QFast-like
    QFastTrie qfast;
    t.tic();
    qfast.build(students, Csize);
    double qf_build_ms = t.toc_ms();
    size_t qf_mem = qfast.approx_bytes();

    cout << "\n=== BUILD ===\n";
    cout << "[FQA]   build: " << fqa_build_ms << " ms | pivots=" << fqa.pivot_idx.size()
         << " | approx mem: " << fqa_mem << " (" << human_bytes(fqa_mem) << ")\n";
    cout << "[Burst] build: " << bt_build_ms << " ms | nodes=" << btrie.nodes.size()
         << " | approx mem: " << bt_mem << " (" << human_bytes(bt_mem) << ")\n";
    cout << "[QFast] build: " << qf_build_ms << " ms | reps=" << qfast.reps.size()
         << " | buckets=" << qfast.buckets.size()
         << " | approx mem: " << qf_mem << " (" << human_bytes(qf_mem) << ")\n";

    // ---------- Prepare queries ----------
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, (int)students.size() - 1);

    vector<double> cg_queries;
    vector<string> name_queries;
    vector<string> reg_queries;

    cg_queries.reserve(Q);
    name_queries.reserve(Q);
    reg_queries.reserve(Q);

    for (int i = 0; i < Q; i++) {
        int idx = dist(rng);
        cg_queries.push_back(students[idx].cgpa);
        name_queries.push_back(students[idx].name);
        reg_queries.push_back(students[idx].reg_no);
    }

    // ---------- Query timing: total ms for Q queries ----------
    cout << "\n=== QUERY (total time for Q=" << Q << ") ===\n";

    volatile size_t sink = 0;

    t.tic();
    for (int i = 0; i < Q; i++) {
        auto res = fqa.query_exact(cg_queries[i], EPS);
        sink += res.size();
    }
    double fqa_query_ms = t.toc_ms();
    cout << "[FQA]   query: " << fqa_query_ms << " ms\n";

    t.tic();
    for (int i = 0; i < Q; i++) {
        auto res = btrie.find_exact(name_queries[i]);
        sink += res.size();
    }
    double bt_query_ms = t.toc_ms();
    cout << "[Burst] query: " << bt_query_ms << " ms\n";

    t.tic();
    for (int i = 0; i < Q; i++) {
        int idx = qfast.find_exact(reg_queries[i]);
        sink += (idx != -1);
    }
    double qf_query_ms = t.toc_ms();
    cout << "[QFast] query: " << qf_query_ms << " ms\n";

    // prevent "unused volatile" warnings
    if (sink == 123456789) cout << "sink\n";

    // ---------- Optional CSV output ----------
    if (!csv.empty()) {
        append_row_csv(csv, "FQA", opt, fqa_build_ms, fqa_query_ms, fqa_mem);
        append_row_csv(csv, "BurstTrie", opt, bt_build_ms, bt_query_ms, bt_mem);
        append_row_csv(csv, "QFastTrie", opt, qf_build_ms, qf_query_ms, qf_mem);
        cout << "\n[OK] Appended 3 rows to: " << csv << "\n";
    }

    return 0;
}
