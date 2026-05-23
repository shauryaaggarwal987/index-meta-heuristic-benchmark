# Results Files

These CSV files are organized to match the final values reported in the paper tables.

## Files

- `baseline_10k.csv`: untuned baseline results for N = 10,000.
- `baseline_20k.csv`: untuned baseline results for N = 20,000.
- `baseline_30k.csv`: untuned baseline results for N = 30,000.
- `baseline_all.csv`: combined baseline results for 10k, 20k, and 30k.
- `tuned_results_30k.csv`: final tuned results for the 30k dataset across all 9 structure-optimizer combinations.
- `consistency_results.csv`: trial-to-trial differences reported in the consistency table.
- `per_metric_rankings_30k.csv`: rankings by build time, query time, and memory.
- `meta_heuristic_benchmark_comparison.xlsx`: workbook version of the final tables.

## Notes

- Time values are reported in milliseconds.
- Baseline values use mean and half-range from two independent trials.
- Tuned values use mean and half-range from two independent trials on the 30k dataset with Q = 2,000 queries.
- Memory values in the paper tables are shown in KB. Baseline CSVs also include MB values for convenience.
- HBSI corresponds to the hash-bucketed string index implementation previously stored in files named `qfast_*.cpp`.
