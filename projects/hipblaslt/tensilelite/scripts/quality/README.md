# tensilelite quality + AI-friendliness metrics

A **report-only**, non-gating measurement for the hipBLASLt tensilelite
`Tensile/` Python tree. It folds two measurements into a single table:

- **complexity / size** via [lizard](https://github.com/terryyin/lizard) —
  function cyclomatic-complexity (CCN) and file-NLOC violator counts
  (Python-only).
- **AI-friendliness** — a stdlib AST scanner (`ai_friendliness.py`) producing 21
  readability signals: file size, deep nesting, wide signatures, swallowed
  errors, broad/bare excepts, mutable default args, global statements, star
  imports, `typing.Any` / `# type: ignore` escape hatches, `print` call sites,
  TODO/FIXME comments, cross-feature imports, duplicated string literals,
  missing seam tests, long lines, etc.

## Running locally

```bash
# Report-only (always exits 0):
python scripts/quality/check_quality.py

# Same report via the manual pre-commit hook:
pre-commit run --config projects/hipblaslt/tensilelite/.pre-commit-config.yaml \
  --hook-stage manual tensilelite-quality --all-files
```

## Targets and enforcement

`quality_targets.json` records the agreed-upon baseline (seeded at the current
measurement). Metrics at or below their target render as `ok`; above target as
`OVER`. In the default report-only mode this is purely informational — the check
**always exits 0**.

Tighten targets deliberately over time to "hold the line". To flip on gating
(exit 1 when a gated metric exceeds its target), run with `--enforce` — e.g. in
CI once the team is ready:

```bash
python scripts/quality/check_quality.py --enforce
```

Re-seed the baseline after an intentional change with `--write-targets`.

## CI

`.github/workflows/hipblaslt-tensilelite-quality.yml` runs this check on PRs
touching `projects/hipblaslt/tensilelite/**` (docs-only PRs excluded) and writes
the table to the run's job summary via `--summary-file "$GITHUB_STEP_SUMMARY"`.
No PR comments are posted.
