# RedRanker AI Project Setup

This is the working project folder for the Redrob candidate ranking challenge.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Included Files

- `job_description.docx`: role details to read first.
- `submission_spec.docx`: submission rules and constraints.
- `redrob_signals_doc.docx`: behavioral signal reference.
- `candidate_schema.json`: schema for candidate records.
- `candidates.jsonl`: full candidate dataset.
- `sample_candidates.json`: small readable candidate sample.
- `sample_submission.csv`: CSV format reference only.
- `submission_metadata_template.yaml`: metadata template for final submission.
- `validate_submission.py`: validator for final submission CSVs.
- `requirements.txt`: Python packages useful for data exploration and project setup.

No ranking implementation has been added yet.
