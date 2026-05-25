# verl-parse

Parse two VERL logs and export step-level metrics to Excel.

## Install

```bash
python3 -m pip install openpyxl
```

### Windows Setup

If `python` is not available in `PATH`, create a virtual environment with a known
Python executable:

```powershell
& 'D:\wecode_build_tools\mingw\bin\python.exe' -m venv .venv
```

Install dependencies with the local proxy:

```powershell
$env:HTTP_PROXY='http://proxysg.huawei.com:8080'
$env:HTTPS_PROXY='http://proxysg.huawei.com:8080'
& '.\.venv\bin\python.exe' -m pip install --upgrade pip openpyxl
```

## Usage

```bash
python3 parse_verl_logs_to_excel.py log1.log log2.log -o result.xlsx
```

On Windows with the virtual environment above:

```powershell
& '.\.venv\bin\python.exe' .\parse_verl_logs_to_excel.py .\log1.log .\log2.log -o .\result.xlsx
```

Optional labels:

```bash
python3 parse_verl_logs_to_excel.py log1.log log2.log -o result.xlsx \
  --label1 baseline --label2 new_run
```

## Output

The workbook contains:

- `metadata`: log source paths, labels, and compare rules
- `log1_metrics`: full parsed metrics for log1
- `log2_metrics`: full parsed metrics for log2
- `summary`: key metrics only
- `compare`: step-level grouped columns for each key metric
- `compare_detail`: full metric-by-metric comparison

In `compare`:

- normal metrics include `log1`, `log2`, `pct_diff`
- performance metrics (`timing_s/*`, `perf/*`) also include `log1/log2` and `log2/log1`
- one overview chart shows percent diff by step
- one trend chart per key metric shows `log1` and `log2` curves over steps
- metric trend charts are placed directly below their corresponding metric columns

## Compare Rules

- `<= 1%`: green
- `1% ~ 5%`: yellow
- `> 5%`: red
- missing or baseline zero: gray

The `compare` sheet chart uses:

- X axis: step number
- Y axis: percent diff vs baseline
- Fixed Y range: `-15%` to `+15%`

## Notes

- ANSI color codes are stripped automatically.
- Metrics are parsed from `key:value` pairs on `step:N` lines.
- `timing_s/logp` accepts aliases such as `timing_s/log_prob` and `timing_s/ref`.
