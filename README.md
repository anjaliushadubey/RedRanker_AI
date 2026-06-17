# RedRanker AI

This is a reproducible baseline project for the Redrob candidate ranking challenge.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Baseline Pipeline

```powershell
python rank.py --candidates .\candidates.jsonl --out .\submission.csv
python validate_submission.py .\submission.csv
```

The pipeline first applies a conservative hard-reject layer for the JD's explicit disqualifiers, zero core relevance, and severe Redrob availability/trust failures. It then scores eligible candidates with rule-based JD features such as title, experience, retrieval/search evidence, ranking/matching evidence, production signal, evaluation signal, Python signal, product-company exposure, and Redrob behavior signals.
It prints a hard-reject audit before scoring, so rejection rules can be tuned safely.

Honeypot detection is handled separately in `src/honeypot.py` and runs before JD fit checks. A honeypot score of 5 or more is hard rejected, 3-4 receives a heavy trap penalty, and 1-2 receives a mild consistency penalty. The checks look for impossible dates, impossible skill durations, too many expert skills with zero/tiny duration, heavy role overlaps, non-tech keyword stuffing, seniority contradictions, unrealistic new-technology duration claims, profile/current-role mismatches, role duration/date mismatches, career duration exceeding claimed experience, explicit text-vs-structured experience contradictions, future education years, future certification years, inverted salary ranges, and last-active-before-signup anomalies. Noisy signals such as salary inversion are weak supporting evidence rather than direct rejection.

The default ranking path uses pure JSONL streaming because benchmarking showed it is faster for this nested, row-wise pipeline. Pandas JSONL chunking is also available for analysis/debugging:

```powershell
python rank.py --candidates .\candidates.jsonl --out .\submission.csv --loader jsonl
python rank.py --candidates .\candidates.jsonl --out .\submission.csv --loader pandas
```

Measured locally: the expanded JSONL streaming pipeline currently runs in about 55 seconds on the full 100k-candidate dataset and remains under the 5-minute CPU-only target. Earlier benchmarking showed Pandas chunking was slower for this nested row-wise architecture, so Pandas remains useful for offline analysis, while the final production run should use `--loader jsonl`.

## Included Files

- `job_description.docx`: role details to read first.
- `submission_spec.docx`: submission rules and constraints.
- `redrob_signals_doc.docx`: behavioral signal reference.
- `candidate_schema.json`: schema for candidate records.
- `candidates.jsonl`: full candidate dataset.
- `sample_candidates.json`: small readable candidate sample.
- `sample_submission.csv`: CSV format reference only.
- `submission_metadata.yaml`: local submission metadata file copied from the template.
- `submission_metadata_template.yaml`: metadata template for final submission.
- `validate_submission.py`: validator for final submission CSVs.
- `requirements.txt`: Python packages useful for data exploration and project setup.

## Project Structure

```text
rank.py
validate_submission.py
requirements.txt
README.md
submission_metadata.yaml
src/
  __init__.py
  config.py
  data_loader.py
  features.py
  honeypot.py
  rejector.py
  scorer.py
  traps.py
  reasoning.py
  writer.py
```

`candidates.jsonl` and generated CSV files are ignored by Git because the dataset is too large for normal GitHub pushes.
