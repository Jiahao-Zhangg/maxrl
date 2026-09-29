# TACO pass@1 and concurrency comparison

`taco_eval.py` evaluates a pinned Qwen3 checkpoint on 100 Easy and 100 Medium
questions sampled from the original TACO training split to compare training
difficulty. Retain raw training questions without SPJ filtering, as requested;
the official SPJ annotations describe only test indices. Training-pool results
must be labeled separately from held-out TACO test benchmark scores.
Before sampling, exclude records without usable nonempty, paired input/output test
lists so each difficulty has 100 gradeable questions. Preserve original global row
indices, record exclusions and reasons, and never filter by solution performance.
A run directory supplies `plan.json`,
frozen questions and tokenizer IDs, model checksums, an independent Python
environment, and an unmodified snapshot of the official FlagOpen/TACO grader.

Use the native thinking chat template, one response per question, a 32,768-token
output cap, temperature 0.6, top-p 0.95 and top-k 20. Extract Python code only from
the response after `</think>`; a missing closing tag fails. No extra EOS check is
applied. Generated code runs in isolated user, mount, PID and network namespaces;
only the environment, official grader and current input are mounted. The official
per-test time limit and result comparisons stay unchanged. Negative error and
timeout codes are failures. Dataset defects are recorded without selecting or
filtering questions by reference-solution or model performance. The SPJ list covers
the test split and must not be applied to train indices.

For the concurrency experiment, each of four physical GPUs receives a fixed,
balanced shard of 50 questions. Each shard runs with `max_num_seqs=16` and `32` on
the same GPU, with the order counterbalanced across GPUs. Sampling seeds, request
order, model, memory budget and all other engine settings are identical. Report
per-condition Easy/Medium/overall pass@1; never pool the conditions into pass@2.
Measure generation time and output-token throughput separately from model loading,
warmup and grading, plus peak GPU memory, observed active sequences and preemptions.
An OOM at 32 remains an explicit failed benchmark outcome; a complete 16 run still
provides the requested accuracy evaluation. Inference headroom does not establish
the memory requirements of training with an actor and optimizer.

The `queue`, `run` and `worker` commands require `--output-root`; workers also take
`--rank` and `--concurrency`. Durable per-shard receipts and per-response grades
support recovery. Controllers verify frozen inputs and their original compute
allocation. GPU launches wait for the preceding stage's complete audit, allocation
locks and idle GPUs. Control records are mirrored to local scratch and retry
filesystem-space errors.

`taco_priority_queue.py` wraps an existing frozen compression-budget queue. It
preserves the running stage and adds a dependency gate before the specified next
stage. Install the wrapper by replacing only identified queue/supervisor processes;
running evaluation workers must keep their original process identities. Keep the
original budget plan and runtime unchanged. The TACO stage must finish its audit
before the original queue resumes.

Results are written to `report/README.md`, `report/metrics.json` and
`report/audit.json`; the run directory retains exact questions, responses, grading
outcomes, checksums and the official grader revision.
