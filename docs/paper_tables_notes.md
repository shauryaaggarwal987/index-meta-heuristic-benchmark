# Paper Tables and Reproducibility Notes

This document maps the main paper tables to the repository files needed to reproduce or verify them.

## Experimental scope

The paper evaluates three attribute-specific index structures:

| Structure | Indexed attribute | Query type | Tuned parameter |
|---|---|---|---:|
| FQA | `cgpa` | hit-only numeric exact-match/proximity-style lookup | `K` |
| Burst Trie | `name` | hit-only string exact-match lookup | `THR` |
| HBSI | `reg_no` | hit-only hash-based exact-match lookup | `C` |

The structures solve different retrieval tasks. Therefore, cross-structure results should be interpreted as attribute-specific deployment guidance, not as a universal ranking of interchangeable data structures.

---

## Dataset construction

Generated file:

```text
data/students_100k.csv
```

Generated using:

```bash
python data/generate_students_100k.py --input NationalNames.csv --output data/students_100k.csv --n 100000 --seed 42
```

Output schema:

```text
name,reg_no,cgpa
```

Notes:

- `name` is taken from the Kaggle National Names source dataset.
- `reg_no` is generated as `REG000001`, `REG000002`, ...
- `cgpa` is synthetic and is produced by log-scaling the source `Count` field into `[0, 10]`.
- The synthetic CGPA field is intended only to provide a controlled numeric lookup workload for FQA.

---

## Baseline parameter table

Baseline parameters used in the paper:

| Structure | Parameter | Default value |
|---|---:|---:|
| FQA | `K` | 16 |
| Burst Trie | `THR` | 32 |
| HBSI | `C` | 64 |

Relevant file:

```text
src/benchmarking_code_new.cpp
```

Representative command:

```bash
g++ -O2 -std=gnu++17 src/benchmarking_code_new.cpp -o bench
./bench --data=data/students_100k.csv --n=30000 --opt=Baseline --csv=results/baseline_30k.csv
```

---

## Baseline performance table

Paper table: baseline build time, query time, and estimated index memory across dataset sizes.

Relevant commands:

```bash
./bench --data=data/students_100k.csv --n=10000 --opt=Baseline --csv=results/baseline_10k.csv
./bench --data=data/students_100k.csv --n=20000 --opt=Baseline --csv=results/baseline_20k.csv
./bench --data=data/students_100k.csv --n=30000 --opt=Baseline --csv=results/baseline_30k.csv
```

Notes:

- Query count defaults to `Q = 2000` unless overridden.
- Memory is recorded by the benchmark in MB and converted to KB for paper tables.
- Memory values represent estimated index memory, not full process resident memory.

---

## Tuned optimizer results table

Paper tables: tuned build time, tuned query time, tuned memory footprint, and optimized parameter ranges.

Relevant files:

```text
src/fqa_abc.cpp
src/fqa_pso.cpp
src/fqa_gwo.cpp
src/burst_abc_fixed.cpp
src/burst_pso_fixed.cpp
src/burst_gwo_fixed.cpp
src/hbsi_abc.cpp
src/hbsi_pso.cpp
src/hbsi_gwo.cpp
```

Each file performs a standalone per-structure optimizer run. The final methodology uses nine independent runs:

```text
3 structures × 3 optimizers = 9 independent optimization runs
```

Each run tunes only one structure-specific parameter:

- FQA tunes `K`.
- Burst Trie tunes `THR`.
- HBSI tunes `C`.

---

## Per-metric rankings table

Paper table: ranking of all nine structure-optimizer combinations by build time, query time, and estimated memory.

Source artifacts:

```text
results/tuned_results_30k.csv
results/meta_heuristic_benchmark_comparison.xlsx
```

Interpretation note:

The rankings show that structure choice dominates optimizer choice. HBSI configurations lead the query-time ranking, while Burst Trie configurations lead the build-time and memory rankings.

---

## Trial-to-trial consistency table

Paper table: absolute differences between two independent trials for each structure-optimizer combination.

Source artifacts:

```text
results/tuned_results_30k.csv
results/meta_heuristic_benchmark_comparison.xlsx
```

Interpretation note:

Two trials provide preliminary repeatability evidence only. They do not support formal confidence intervals or hypothesis testing.

---

## Important caveats preserved in the paper

The final paper explicitly notes the following methodological scope limits:

1. The structures operate on different attributes and query types.
2. The query workload is hit-only exact-match lookup.
3. HBSI is a hash-bucketed string index inspired by Q-Fast bucketing, not Willard's original Q-Fast Trie.
4. The fitness function is weighted-sum scalarization, not true Pareto-front optimization.
5. Memory values are estimated index memory footprints.
6. Very small query times should be interpreted as controlled microbenchmark timings.
7. The benchmark uses two independent trials per configuration, so statistical claims are preliminary.
