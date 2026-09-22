# Five-model math evaluation — H100 continuation

**700 of 700 budget/seed points complete.**

This snapshot combines 541 published GH200 points with 159 new H100 points. Only missing points were generated. All new raw response/token ledgers passed checksum verification and an independent budget/counter audit.

## Results

- [Tables and figures](results.md)
- [All per-seed points, with source-run labels](all_points.csv)
- [All 700 raw trajectories: downloads and integrity](RAW_TRAJECTORIES.md)
- [Point receipts](point_receipts.csv)
- [Continuation ledger audit](continuation_ledger_audit.json)

## Provenance and limitations

The original 541-point snapshot retains its published metrics and provenance. Its raw trajectories were subsequently restored from the verified GitHub Release archives; their hashes match the original receipts. The 159 continuation points add the missing Eval3 trajectories. See RAW_TRAJECTORIES.md for the complete 700-point index and archive downloads.

Original hardware: four GH200 GPUs, torch 2.6.0+cu126 and vLLM 0.8.4+cu126. Continuation hardware: eight H100 80GB GPUs, one inference engine per GPU, torch 2.6.0+cu124 and vLLM 0.8.4. Models, datasets, converted model files, tokenizer, seed code, budget engine, grading code, Transformers and Math-Verify versions match exactly. GPU/CUDA differences can still affect floating-point results and sampled responses; this is a mixed-hardware completion, not a claim of bitwise reproduction.

[Baseline manifest](baseline_run_manifest.json) and [continuation manifest](continuation_run_manifest.json) preserve each source. The continuation runner adds point selection; generation, token accounting, seeds and grading are unchanged.

Eval1 is mean@4, not pass@4. Eval2 uses per-question output budgets and stops on first success. Eval3 shares M×B output tokens, skips solved questions, and keeps all M questions in the denominator. Plots show measured means over three seeds, with solved-question counts for each dataset and an equal-weight accuracy Macro Average. For each seed, macro accuracy is the mean of the four dataset solved fractions; its mean and sample standard deviation over seeds are saved in CSV. All datasets and all requested seeds must be complete. Solid lines are log2-budget linear fits; dashed extensions are extrapolations, not measured results.

The styling follows [L1 Figure 2](https://arxiv.org/pdf/2503.04697v2#page=6). CAM - Individual Budget denotes Fixed-N RB L+512; CAM - Shared Budget denotes F-Cov L+512; L1, ER and Base denote the three baselines. These legend names identify models; Eval2 and Eval3 retain their respective evaluation protocols. [Eval2 macro averages](eval2_macro_average.csv) and [Eval3 macro averages](eval3_macro_average.csv) accompany the plots.

This Git snapshot contains result tables, figures and provenance metadata. Generated responses and their original grades/token ledgers are provided separately as linked GitHub Release assets. Model weights, credentials and personal filesystem paths are excluded.
