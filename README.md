# verl-parse

Utilities for crunching VERL training logs into a single Excel workbook.

There are two scripts in this repo:

| Script | What it does |
| --- | --- |
| `parse_verl_logs_to_excel.py` | Side-by-side comparison of two specific logs (one sheet per log + a coloured `compare` sheet). Useful when you have a baseline run and a candidate. |
| `parse_verl_logs_dir.py` | Walks a directory tree, parses every `*.log` it finds, and dumps one workbook with a single `summary` sheet + one full step-by-step sheet per log. The summary sheet is the one to open when you're chasing throughput and MFU across many optimisation runs. |

`gen_test_fixtures.py` is a small helper that produces a few synthetic logs
in `testFold/` so you can see what the parser does without bringing in real
runs.

## Install

```bash
python3 -m pip install openpyxl
```

### Windows Setup

If `python` is not available in `PATH`, create a virtual environment with a
known Python executable:

```powershell
& 'D:\wecode_build_tools\mingw\bin\python.exe' -m venv .venv
```

Install dependencies with the local proxy:

```powershell
$env:HTTP_PROXY='http://proxysg.huawei.com:8080'
$env:HTTPS_PROXY='http://proxysg.huawei.com:8080'
& '.\.venv\bin\python.exe' -m pip install --upgrade pip openpyxl
```

## Two-log compare

```bash
python3 parse_verl_logs_to_excel.py log1.log log2.log -o result.xlsx
python3 parse_verl_logs_to_excel.py log1.log log2.log -o result.xlsx \
  --label1 baseline --label2 new_run
```

The workbook contains `metadata`, one `*_metrics` sheet per log, a `summary`
sheet, a coloured `compare` sheet, and a `compare_detail` sheet. See the
inline `compare` rules in the spreadsheet for the colour thresholds (green
<= 1%, yellow 1-5%, red > 5%, gray = missing).

## Directory scan (summary workbook)

```bash
python3 parse_verl_logs_dir.py testFold/ -o testFold_summary.xlsx
```

`testFold/` is walked recursively and every `*.log` becomes one row in the
output's `summary` sheet (plus its own full-metrics sheet).

### Summary sheet layout

The first column is a `unique_id` (the file's relative path + filename, e.g.
`A5/06-04/cudagraph.log`) followed by, in order:

1. **Identity** -- `label`, `rel_path`, `n_steps`, `first_step`, `last_step`.
2. **Per-step throughputs** -- `step1_throughput` (warmup), the most recent
   10 steps as `stepN_minus_1_throughput` ... `stepN_minus_10_throughput`,
   plus `avg_throughput_excl_step1` (the steady-state average).
3. **Per-stage timings (avg, excluding first step)** -- `perf/total_num_tokens`,
   `perf/mfu/actor*`, and the `timing_s/*` columns you actually care about:
   `gen`, `old_log_prob`, `ref`, `update_actor`, `update_weights`, `step`,
   `adv`, `reward`.
4. **First-step timings** -- the same per-stage values measured at step 1
   so you can read off the warmup overhead.
5. **Hyperparameters that move the needle** -- `actor_tp/pp/cp/ep/etp/sp`,
   `actor_use_remove_padding`, `actor_grad_offload`, `actor_optim_offload`,
   `actor_ppo_*`, `actor_clip_ratio`, `actor_kl_loss_coef`,
   `actor_entropy_coeff`, `actor_lr`, `rollout_tp/dp/expert_parallel_size`,
   `rollout_n`, `rollout_prompt_length`, `rollout_response_length`,
   `rollout_max_num_batched_tokens`, `rollout_max_num_seqs`,
   `rollout_gpu_memory_utilization`, the rollout optimisation flags
   (`enable_chunked_prefill`, `enable_prefix_caching`, `enforce_eager`,
   `use_torch_compile`), `rollout_dtype`, `train_batch_size`, the
   algorithm and engine keys (`algorithm_adv_estimator`,
   `algorithm_use_kl_in_reward`, `model_engine`).

Each log also gets its own sheet (`log_<safe_name>`) with the full
step-by-step metric grid (one row per step, one column per metric).

### Config extraction

The config dump is recovered from the multi-line Hydra `str(dict)` at the
top of each log.  If a `UserWarning` or a stray Python statement is
interleaved with the dict (as happens on some real logs), the parser drops
those lines and re-tries `ast.literal_eval`.  Logs that genuinely lack a
config header (e.g. a hand-trimmed snippet) just show `n/a` for the
hyperparameter columns; the throughput/timing columns still fill in.

### Generating test logs

```bash
python3 gen_test_fixtures.py
```

This writes eight synthetic logs (8-40 steps each, varied parallel and
optimisation settings) into `testFold/`.  Use them to sanity-check the
parser or as a starting point for your own fixtures.

## Notes

- ANSI escape sequences and the literal `[36m(...)[0m` log prefix are
  stripped automatically.
- Step metrics are parsed from `key:value` pairs on lines that contain
  `step:N`.
- `timing_s/logp` accepts aliases such as `timing_s/log_prob` and
  `timing_s/ref`.
- `ast.literal_eval` is used to recover the printed config; if the
  dict is broken up by warnings the parser strips those lines first.
