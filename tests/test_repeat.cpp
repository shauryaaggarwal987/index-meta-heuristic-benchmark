#include <bits/stdc++.h>
using namespace std;
#define main baseline_main
#include "benchmarking_code.cpp"
#undef main

namespace memtrack {
    struct Header { size_t size; void* raw; };
    static atomic<size_t> g_live{0}, g_peak{0}, g_total{0};
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
        size_t pad = (alignment==0?0:(alignment-(addr%alignment))%alignment);
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
        void reset(){ base_live=g_live.load(); base_total=g_total.load(); base_peak=g_peak.load(); }
        size_t live_delta() const { size_t c=g_live.load(); return c>=base_live?c-base_live:0; }
        size_t total_delta() const { size_t c=g_total.load(); return c>=base_total?c-base_total:0; }
        size_t peak_delta() const { size_t c=g_peak.load(); return c>base_peak?c-base_peak:0; }
    };
}
void* operator new(size_t sz){ return ::memtrack::alloc_block(sz, alignof(max_align_t)); }
void operator delete(void* p) noexcept { ::memtrack::free_block(p); }
void* operator new[](size_t sz){ return ::memtrack::alloc_block(sz, alignof(max_align_t)); }
void operator delete[](void* p) noexcept { ::memtrack::free_block(p); }
void operator delete(void* p, size_t) noexcept { ::memtrack::free_block(p); }
void operator delete[](void* p, size_t) noexcept { ::memtrack::free_block(p); }

using clk = chrono::steady_clock;
static inline long long span_ns(const clk::time_point& a, const clk::time_point& b){
    return chrono::duration_cast<chrono::nanoseconds>(b-a).count();
}
static inline void do_not_optimize_away(const volatile void* p){
    asm volatile("" : : "g"(p) : "memory");
}
template<class T>
auto try_approx_bytes(T& t, int) -> decltype(t.approx_bytes(), size_t{}) { return t.approx_bytes(); }
template<class T>
size_t try_approx_bytes(T&, long){ return 0; }

int main(){
    auto students = load_students_csv("students_10k.csv");
    fprintf(stderr, "Loaded: %d students\n", (int)students.size());
    
    static volatile size_t sink = 0;
    
    // Simulate what evaluate_config does
    for (int rep = 0; rep < 5; ++rep) {
        FQAIndex tmp;
        tmp.build(students, 16);
        sink ^= try_approx_bytes(tmp, 0) + 1;
        do_not_optimize_away(&tmp);
    }
    fprintf(stderr, "FQA repeated builds done\n");
    
    for (int rep = 0; rep < 5; ++rep) {
        BurstTrie tmp;
        tmp.build(students, 32);
        sink ^= try_approx_bytes(tmp, 0) + 3;
        do_not_optimize_away(&tmp);
    }
    fprintf(stderr, "BurstTrie repeated builds done\n");
    
    for (int rep = 0; rep < 5; ++rep) {
        QFastTrie tmp;
        tmp.build(students, 64);
        do_not_optimize_away(&tmp);
    }
    fprintf(stderr, "QFastTrie repeated builds done\n");
    fprintf(stderr, "All done! sink=%zu\n", (size_t)sink);
    return 0;
}
