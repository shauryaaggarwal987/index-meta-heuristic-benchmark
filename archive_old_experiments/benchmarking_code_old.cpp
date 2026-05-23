#include <bits/stdc++.h>
using namespace std;

/*
  Benchmarking + Query Timing + Memory Estimation for three indices over a student dataset.
  Payload per record (STRICTLY 3 fields): {name, reg_no, cgpa}

  Indexes:
  1) FQA (Fixed Queries Array) on CGPA
     Build:  O(n*k + n log n)
     Query:  ~O(log n + w)  (w = tiny local scan)
     Space:  O(n*k + n)

  2) Burst Trie on Name
     Build:  O(Σ|name|)
     Query:  exact ~ O(|q| + occ)
     Space:  O(nodes + Σ|bucket|)

  3) Q-Fast-like Trie on hashed Reg No
     Build:  O(n log n + |S*| * h_bytes)
     Query:  O(log |S*| + log c)  (implemented as reps predecessor + bucket search)
     Space:  O(n + |S*| + buckets + upper)
*/

// --------------------- Dataset record ---------------------
struct Student {
    string name;
    string reg_no;
    double cgpa;
};

// <<< include your generated header just after Student >>>
#include "students_data.hpp" // inline vector<Student> load_students_static()

// --------------------- Misc utils ---------------------
static inline string trim(const string &s) {
    size_t i = 0, j = s.size();
    while (i < j && isspace((unsigned char)s[i])) ++i;
    while (j > i && isspace((unsigned char)s[j-1])) --j;
    return s.substr(i, j - i);
}
static vector<string> split_csv_line(const string& line) {
    vector<string> out;
    string cur;
    bool inq = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '\"') inq = !inq;
        else if (c == ',' && !inq) { out.push_back(trim(cur)); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(trim(cur));
    return out;
}
static vector<Student> load_students_csv(const string& path) {
    vector<Student> v;
    ifstream fin(path);
    if (!fin.is_open()) return v;
    string line; bool header_unknown = true;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto cols = split_csv_line(line);
        if (cols.size() < 3) continue;
        if (header_unknown) {
            try { (void)stod(cols[2]); }
            catch (...) { header_unknown = false; continue; }
        }
        Student s;
        s.name   = cols[0];
        s.reg_no = cols[1];
        try { s.cgpa = stod(cols[2]); } catch (...) { continue; }
        v.push_back(s);
    }
    return v;
}
static string human_bytes(size_t x) {
    static const char* S[] = {"B","KB","MB","GB","TB"};
    int i = 0; double d = (double)x;
    while (d >= 1024.0 && i < 4) { d /= 1024.0; ++i; }
    ostringstream os; os.setf(std::ios::fixed);
    os<<setprecision(d>=100?0:(d>=10?1:2))<<d<<" "<<S[i];
    return os.str();
}

// --------------------- 1) FQA over CGPA ---------------------
struct FQAIndex {
    const vector<Student>* students = nullptr;
    vector<int> pivot_idx;                 // pivot indices
    vector<int> order;                     // permutation
    vector< array<float,32> > coords_small;// coordinates to <=32 pivots
    int k = 0;

    static vector<int> choose_pivots(const vector<Student>& st, int k) {
        int n = (int)st.size();
        vector<int> piv;
        if (n == 0 || k <= 0) return piv;
        vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b){ return st[a].cgpa < st[b].cgpa; });
        piv.push_back(idx[n/2]);
        auto dist = [&](int i, int p){ return fabs(st[i].cgpa - st[p].cgpa); };
        while ((int)piv.size() < k) {
            int best = -1; double bestd = -1;
            for (int i = 0; i < n; ++i) {
                double dmin = 1e100;
                for (int p : piv) dmin = min(dmin, dist(i,p));
                if (dmin > bestd) { bestd = dmin; best = i; }
            }
            if (best == -1) break;
            piv.push_back(best);
        }
        return piv;
    }

    void build(const vector<Student>& st, int k_pivots = 16) {
        students = &st;
        int n = (int)st.size();
        k = max(1, min(32, min(k_pivots, max(1, n))));
        pivot_idx = choose_pivots(st, k);
        order.resize(n); iota(order.begin(), order.end(), 0);
        coords_small.assign(n, array<float,32>{});
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                coords_small[i][j] = (float)fabs(st[i].cgpa - st[pivot_idx[j]].cgpa);
            }
        }
        sort(order.begin(), order.end(), [&](int a, int b){
            for (int j = 0; j < k; ++j) {
                if (coords_small[a][j] < coords_small[b][j]) return true;
                if (coords_small[a][j] > coords_small[b][j]) return false;
            }
            if (st[a].reg_no != st[b].reg_no) return st[a].reg_no < st[b].reg_no;
            return st[a].name < st[b].name;
        });
    }

    array<float,32> target_coords(double cg) const {
        array<float,32> t{};
        for (int j=0;j<k;++j) t[j] = (float)fabs(cg - students->at(pivot_idx[j]).cgpa);
        return t;
    }
    bool coords_lt(int idx, const array<float,32>& T) const {
        const auto &A = coords_small[idx];
        for (int j=0;j<k;++j) {
            if (A[j] < T[j]) return true;
            if (A[j] > T[j]) return false;
        }
        return false;
    }

    vector<int> query_exact(double cg, double eps=1e-9) const {
        if (!students) return {};
        int n = (int)order.size();
        if (n==0) return {};
        auto T = target_coords(cg);
        int lo=0, hi=n;
        while (lo<hi) {
            int mid=(lo+hi)/2;
            int idx = order[mid];
            if (coords_lt(idx, T)) lo=mid+1; else hi=mid;
        }
        vector<int> ans;
        for (int dir=-1; dir<=1; dir+=2) {
            int i = lo + (dir==1?0:-1);
            while (i>=0 && i<n) {
                int idx = order[i];
                double g = students->at(idx).cgpa;
                if (fabs(g-cg) <= eps) ans.push_back(idx);
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
        unordered_map<unsigned char, int> nxt; // access node transitions
        vector<int> bucket;                    // leaf container: indices into students
    };

    const vector<Student>* students = nullptr;
    vector<Node> nodes;
    int threshold;

    BurstTrie(int threshold_=32) : threshold(max(4,threshold_)) {
        nodes.reserve(4096);
        nodes.push_back(Node()); // root
    }

    void _burst(int u, const vector<int>& ids, size_t depth) {
        nodes[u].isLeaf = false;
        nodes[u].nxt.clear();
        for (int id : ids) {
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
                _burst(v, ids2, depth+1);
            }
        }
    }

    void insert(const vector<Student>& st, int id) {
        if (!students) students = &st;
        int u = 0; size_t d = 0;
        const string& s = st[id].name;
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
                        _burst(u, ids, d+1);
                    }
                    return;
                } else { u = it->second; ++d; }
            }
        }
    }

    void build(const vector<Student>& st, int threshold_=32) {
        students = &st;
        threshold = max(4, threshold_);
        nodes.clear();
        nodes.push_back(Node()); // root
        for (int i = 0; i < (int)st.size(); ++i) insert(st, i);
    }

    vector<int> find_exact(const string& name) const {
        int u = 0; size_t d = 0;
        while (true) {
            if (nodes[u].isLeaf) {
                vector<int> ans;
                for (int id : nodes[u].bucket)
                    if (students->at(id).name == name) ans.push_back(id);
                return ans;
            } else {
                unsigned char c = (d < name.size()) ? (unsigned char)name[d] : (unsigned char)0;
                auto it = nodes[u].nxt.find(c);
                if (it == nodes[u].nxt.end()) return {};
                u = it->second; ++d;
            }
        }
    }

    static size_t approx_unordered_map_bytes(const unordered_map<unsigned char,int>& m) {
        size_t pairs = m.size() * sizeof(pair<const unsigned char,int>);
        size_t buckets = m.bucket_count() * sizeof(void*);
        return pairs + buckets;
    }
    size_t approx_bytes() const {
        size_t bytes = sizeof(Node) * nodes.capacity();
        for (const auto& nd : nodes) {
            bytes += nd.bucket.capacity() * sizeof(int);
            if (!nd.isLeaf) bytes += approx_unordered_map_bytes(nd.nxt);
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
    static uint64_t key_from_reg(const string& s) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (unsigned char c : s) { h ^= c; h *= 0x100000001b3ULL; }
        return splitmix64(h);
    }

    struct UpperNode {
        vector<uint8_t> present;
        vector<int> child;
        bool isLeaf = false;
        int leaf_index = -1;
    };
    struct Rep { uint64_t key; int bucket_id; };
    struct Bucket { vector<pair<uint64_t,int>> items; };

    const vector<Student>* students = nullptr;
    vector<uint64_t> keys;
    vector<int> order_by_key;
    vector<Rep> reps;
    vector<Bucket> buckets;
    vector<UpperNode> up;
    int root = -1;
    int h_bytes = 8;
    int c = 64;

    // FIXED: avoid returning through references after up.push_back() (which may reallocate)
    int ensure_child(int u, uint8_t d) {
        vector<uint8_t> &P = up[u].present;
        vector<int>     &C = up[u].child;
        auto it = lower_bound(P.begin(), P.end(), d);
        if (it != P.end() && *it == d) {
            int pos = int(it - P.begin());
            if (pos >= (int)C.size() || C[pos] == -1) {
                if ((int)C.size() < (int)P.size()) C.resize(P.size(), -1);
                int childIdx = (int)up.size();
                C[pos] = childIdx;
                up.push_back(UpperNode());
                return childIdx;
            }
            return C[pos];
        } else {
            int pos = int(it - P.begin());
            P.insert(it, d);
            // ensure C has at least size P.size() - 1 before we insert at position pos
            if ((int)C.size() < pos) C.resize(pos, -1);
            int childIdx = (int)up.size();
            if ((int)C.size() < (int)P.size()) {
                C.insert(C.begin() + pos, childIdx);
            } else {
                // size equal already (should rarely happen here), just assign
                C[pos] = childIdx;
            }
            up.push_back(UpperNode());
            return childIdx;
        }
    }

    void insert_rep(uint64_t key, int rep_idx) {
        int u = root;
        for (int depth = 0; depth < h_bytes; ++depth) {
            uint8_t d = (uint8_t)(key >> (8*(h_bytes-1-depth)));
            int v = ensure_child(u, d);
            u = v;
        }
        up[u].isLeaf = true;
        up[u].leaf_index = rep_idx;
    }

    void build(const vector<Student>& st, int c_bucket = 64) {
        students = &st;
        c = max(8, c_bucket);
        int n = (int)st.size();
        keys.resize(n);
        for (int i = 0; i < n; ++i) keys[i] = key_from_reg(st[i].reg_no);
        order_by_key.resize(n);
        iota(order_by_key.begin(), order_by_key.end(), 0);
        sort(order_by_key.begin(), order_by_key.end(),
            [&](int a, int b){ return (keys[a] != keys[b]) ? keys[a] < keys[b] : st[a].reg_no < st[b].reg_no; });

        reps.clear(); buckets.clear();
        for (int i = 0; i < n; i += c) {
            int first = i, last = min(n, i + c) - 1;
            int rep_id = order_by_key[first];
            uint64_t krep = keys[rep_id];
            int bucket_id = (int)buckets.size();
            reps.push_back(Rep{krep, bucket_id});
            buckets.push_back(Bucket{});
            auto &B = buckets.back().items;
            B.reserve(last-first+1);
            for (int t = first; t <= last; ++t) {
                int sid = order_by_key[t];
                B.emplace_back(keys[sid], sid);
            }
        }

        up.clear(); up.push_back(UpperNode()); root = 0;
        up[0].child.clear(); up[0].present.clear();
        // Reserve to reduce reallocations (optional safety)
        up.reserve((size_t)reps.size() * (size_t)h_bytes + 4);
        for (int r = 0; r < (int)reps.size(); ++r) insert_rep(reps[r].key, r);

        // --- Sanity guards (avoid malformed structures)
        if (reps.empty()) {
            reps.push_back(Rep{ keys[ order_by_key[0] ], 0 });
            if (buckets.empty()) buckets.push_back(Bucket{});
        }
        if (up.empty()) { up.push_back(UpperNode()); root = 0; }
        for (auto &nd : up) {
            if (nd.child.size() < nd.present.size()) nd.child.resize(nd.present.size(), -1);
            if (nd.child.size() > nd.present.size()) nd.child.erase(nd.child.begin() + nd.present.size(), nd.child.end());
        }
    }

    // Safe query: reps predecessor + bucket search only
    int find_exact(const string& reg_no) const {
        if (!students || students->empty() || reps.empty() || buckets.empty()) return -1;

        uint64_t kq = key_from_reg(reg_no);

        // predecessor on reps: largest rep.key <= kq
        int R = int(upper_bound(reps.begin(), reps.end(), kq,
                [](uint64_t val, const Rep& r){ return val < r.key; }) - reps.begin()) - 1;
        if (R < 0) R = 0;
        if (R >= (int)reps.size()) R = (int)reps.size() - 1;

        const auto& B = buckets[ reps[R].bucket_id ].items;

        auto cmp = [](const pair<uint64_t,int>& a, const pair<uint64_t,int>& b){
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        };
        auto it_lb = lower_bound(B.begin(), B.end(), make_pair(kq, -1), cmp);

        // forward scan over equal keys
        for (auto jt = it_lb; jt != B.end() && jt->first == kq; ++jt) {
            int sid = jt->second;
            if (students->at(sid).reg_no == reg_no) return sid;
        }
        // backward scan over equal keys just before lb
        if (it_lb != B.begin()) {
            auto jt = it_lb; --jt;
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
        bytes += up.capacity() * sizeof(UpperNode);
        for (const auto& nd : up) {
            bytes += nd.present.capacity() * sizeof(uint8_t);
            bytes += nd.child.capacity() * sizeof(int);
        }
        for (const auto& B : buckets) {
            bytes += B.items.capacity() * sizeof(pair<uint64_t,int>);
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

template<class F>
vector<double> time_many(F&& fn, int Q) {
    vector<double> us; us.reserve(Q);
    for (int i=0;i<Q;++i) {
        auto t0 = chrono::high_resolution_clock::now();
        fn(i);
        auto t1 = chrono::high_resolution_clock::now();
        double dt = chrono::duration<double, micro>(t1 - t0).count();
        us.push_back(dt);
    }
    return us;
}
static void print_stats(const string& what, const vector<double>& us) {
    if (us.empty()) return;
    vector<double> v = us; sort(v.begin(), v.end());
    double avg = accumulate(v.begin(), v.end(), 0.0)/v.size();
    auto pct = [&](double p){ size_t i = (size_t)floor(p*(v.size()-1)+0.5); return v[min(i, v.size()-1)]; };
    cout << what << " | avg " << avg << " us"
         << " | p50 " << pct(0.50) << " | p95 " << pct(0.95)
         << " | p99 " << pct(0.99) << " | max " << v.back() << "\n";
}

static map<string,string> parse_args(int argc, char** argv) {
    map<string,string> m;
    for (int i=1;i<argc;++i) {
        string s(argv[i]);
        if (s.rfind("--",0)==0) {
            auto eq = s.find('=');
            if (eq == string::npos) m[s.substr(2)] = "1";
            else m[s.substr(2,eq-2)] = s.substr(eq+1);
        }
    }
    return m;
}

// --------------------- main ---------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args  = parse_args(argc, argv);
    int K      = args.count("k")   ? stoi(args["k"])   : 16;   // FQA pivots
    int THR    = args.count("thr") ? stoi(args["thr"]) : 32;   // Burst threshold
    int Csize  = args.count("c")   ? stoi(args["c"])   : 64;   // QFast bucket size
    int Q      = args.count("Q")   ? stoi(args["Q"])   : 2000; // queries per index
    uint32_t seed = args.count("seed")? (uint32_t)stoul(args["seed"]) : 42u;
    double EPS = args.count("eps") ? stod(args["eps"]) : 1e-9;

    // Use the compiled-in dataset from students_data.hpp
    vector<Student> students = load_students_static();
    if (students.empty()) {
        cerr << "Static dataset is empty. Ensure students_data.hpp is included after struct Student.\n";
        return 1;
    }
    cout << "Loaded students: " << students.size() << "\n";

    // --- Build FQA
    FQAIndex fqa;
    Timer t;
    t.tic(); fqa.build(students, K); double t_fqa = t.toc_ms();
    size_t m_fqa = fqa.approx_bytes();
    cout << "[FQA]   build: " << t_fqa << " ms"
         << " | pivots=" << fqa.pivot_idx.size()
         << " | approx mem: " << m_fqa << " (" << human_bytes(m_fqa) << ")\n";

    // --- Build Burst Trie
    BurstTrie btrie(THR);
    t.tic(); btrie.build(students, THR); double t_bt = t.toc_ms();
    size_t m_bt = btrie.approx_bytes();
    cout << "[Burst] build: " << t_bt << " ms"
         << " | nodes=" << btrie.nodes.size()
         << " | approx mem: " << m_bt << " (" << human_bytes(m_bt) << ")\n";
    if (btrie.nodes.empty()) { cerr << "[Burst] Build produced no nodes — aborting.\n"; return 1; }

    // --- Build QFast
    QFastTrie qfast;
    t.tic(); qfast.build(students, Csize); double t_qf = t.toc_ms();
    size_t m_qf = qfast.approx_bytes();
    cout << "[QFast] build: " << t_qf << " ms"
         << " | reps=" << qfast.reps.size() << " buckets=" << qfast.buckets.size()
         << " | approx mem: " << m_qf << " (" << human_bytes(m_qf) << ")\n";
    if (qfast.reps.empty() || qfast.buckets.empty()) { cerr << "[QFast] Build invalid — aborting.\n"; return 1; }

    // Prepare random queries sampled from the dataset
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, (int)students.size()-1);

    // --- Queries for FQA: exact cgpa matches
    vector<double> cg_queries; cg_queries.reserve(Q);
    for (int i=0;i<Q;++i) cg_queries.push_back(students[ dist(rng) ].cgpa);

    auto fqa_times = time_many([&](int i){
        volatile auto res = fqa.query_exact(cg_queries[i], EPS);
        (void)res;
    }, Q);
    print_stats("[FQA]   query (exact cgpa)", fqa_times);

    // --- Queries for Burst Trie: exact name matches
    vector<string> name_queries; name_queries.reserve(Q);
    for (int i=0;i<Q;++i) name_queries.push_back(students[ dist(rng) ].name);

    auto bt_times = time_many([&](int i){
        volatile auto res = btrie.find_exact(name_queries[i]);
        (void)res;
    }, Q);
    print_stats("[Burst] query (exact name)", bt_times);

    // --- Queries for QFast: exact reg_no matches
    vector<string> reg_queries; reg_queries.reserve(Q);
    for (int i=0;i<Q;++i) reg_queries.push_back(students[ dist(rng) ].reg_no);

    auto qf_times = time_many([&](int i){
        volatile int idx = qfast.find_exact(reg_queries[i]);
        (void)idx;
    }, Q);
    print_stats("[QFast] query (exact reg_no)", qf_times);

    // Sanity: self-lookup
    {
        int ok = 0, tries = min(100, (int)students.size());
        for (int i=0;i<tries;++i) {
            int s = dist(rng);
            auto v1 = fqa.query_exact(students[s].cgpa, EPS);
            bool hit1 = any_of(v1.begin(), v1.end(), [&](int x){return x==s;});
            auto v2 = btrie.find_exact(students[s].name);
            bool hit2 = any_of(v2.begin(), v2.end(), [&](int x){return x==s;});
            int v3 = qfast.find_exact(students[s].reg_no);
            bool hit3 = (v3==s);
            ok += (hit1 && hit2 && hit3);
        }
        cout << "Sanity self-lookup hits (out of " << tries << "): " << ok << "\n";
    }

    // Complexity summary for report
    cout << "\n=== Theoretical Time/Space Complexity Summary ===\n";
    cout << "FQA (k pivots): Build O(n*k + n log n); Query ~ O(log n + w); Space ~ O(n*k + n)\n";
    cout << "Burst Trie:     Build O(Σ|name|); Query(exact) ~ O(|q| + occ); Space ~ O(nodes + Σ|bucket|)\n";
    cout << "Q-Fast Trie:    Build O(n log n + |S*|*h_bytes); Query ~ O(log |S*| + log c); Space ~ O(n + |S*| + buckets + upper)\n";
    cout << "Notes: h_bytes=8 for 64-bit keys; c is bucket target size (~" << Csize << "). Memory figures are approximate.\n";

    return 0;
}
