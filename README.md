# verl-parse

Parse two VERL logs and export step-level metrics to Excel.

## Install

```bash
python3 -m pip install openpyxl
```

## Usage

```bash
python3 parse_verl_logs_to_excel.py log1.log log2.log -o result.xlsx
```

Optional labels:

```bash
python3 parse_verl_logs_to_excel.py log1.log log2.log -o result.xlsx \
  --label1 baseline --label2 new_run
```

## Output

The workbook contains:

- `log1_metrics`: full parsed metrics for log1
- `log2_metrics`: full parsed metrics for log2
- `summary`: key metrics only
- `compare`: step x key-metric percent-diff matrix with color highlighting and chart
- `compare_detail`: full metric-by-metric comparison

## Compare Rules

- `<= 1%`: green
- `1% ~ 5%`: yellow
- `> 5%`: red
- missing or baseline zero: gray

## Notes

- ANSI color codes are stripped automatically.
- Metrics are parsed from `key:value` pairs on `step:N` lines.
- `timing_s/logp` accepts aliases such as `timing_s/log_prob` and `timing_s/ref`.
