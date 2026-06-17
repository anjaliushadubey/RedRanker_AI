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

The first version is intentionally boring: it scores every candidate with a small experience + Python skill baseline, writes the top 100, and validates the CSV format. Improve scoring only after this pipeline stays green.

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
  scorer.py
  traps.py
  reasoning.py
  writer.py
```

`candidates.jsonl` and generated CSV files are ignored by Git because the dataset is too large for normal GitHub pushes.
