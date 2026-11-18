# English to Chinese MARC pilot

中英文评论星级分类初步复现，涉及论文1、2、3、6。This repository contains a completed single-seed pilot, not a full numerical replication of all four papers.

## Results

| Model | English accuracy | Chinese accuracy | Chinese Macro F1 | Chinese MAE |
|---|---:|---:|---:|---:|
| mBERT | 59.50% | 35.64% | 0.3299 | 0.9536 |
| XLM-R Base | 61.88% | 49.90% | 0.4951 | 0.6574 |
| XLM-R plus DANN | 61.84% | 52.28% | 0.5211 | 0.6128 |

The same stratified 10,000 labelled English reviews were used by all models. DANN additionally used 10,000 Chinese texts without rating labels. English validation was split into 2,500 selection and 2,500 calibration examples. English and Chinese full test sets each contain 5,000 reviews. Seed 42, two epochs, five star classes, RTX 4060 Laptop 8 GB. Temperature scaling was fitted on English only and did not change predicted classes. See [protocol](docs/PROTOCOL.md), [result notes](docs/RESULTS_README.md), [all results](reports/marc_en_zh_pilot/all_results.csv) and the English/Chinese Word reports in `reports/`.

## Layout

```text
src/                 Original experiment implementation
configs/             Executed pilot and unexecuted full-budget configuration
scripts/             Setup, pinned downloads, restoration and checks
tests/               Metrics, calibration and gradient reversal tests
data/provenance/     Download URLs, revisions, checksums and split audits
data/splits/         Exact sampled review IDs without review texts
models/pretrained/   Encoder configs, tokenizers and upstream model cards
models/finetuned/    Architecture metadata without checkpoint tensors
results/             Metrics, temperatures, histories, logits and prediction IDs
reports/             HTML, figures, per-paper CSVs and bilingual Word reports
environment/         Historical dependency lock and hardware record
references/          Paper links and asset attribution
docs/                Protocol, results, upload and checkpoint instructions
```

Raw reviews, processed texts, token caches, weight tensors, PDFs and virtual environments are NOT included in the public repository. In the local delivery they are preserved in the adjacent `../local_assets/`. This repository is not an offline self-contained model bundle. Public clones cannot recover the exact trained checkpoints from download links: they must obtain a separately shared checkpoint or retrain.

## View the completed study

Open `reports/marc_en_zh_pilot/index.html` locally. CSVs `paper1_results.csv`, `paper2_results.csv`, `paper3_results.csv`, and `paper6_results.csv` separate each paper's contribution. These files are results of this pilot, not the papers' published numbers.

## Installation and reproduction

Original environment: Windows, Python 3.12, PyTorch 2.6.0 with CUDA 12.4. Training code currently requires a compatible NVIDIA CUDA GPU for FP16. CPU-only training and other platforms have not been validated.

```powershell
# Run from this repository directory. Uses a repo-local .venv.
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Choose one asset source:

```powershell
# Current local delivery: recover all original assets, including trained weights.
.\.venv\Scripts\python.exe scripts/restore_local_assets.py

# Fresh public clone: download exact recorded upstream data and BASE models.
# Check upstream terms and accessibility first. Network availability is not guaranteed.
.\.venv\Scripts\python.exe scripts/download_assets.py
```

The downloader uses `data/provenance/downloads.json`, never the latest branch. Every downloaded file is checked against its saved SHA256. It does not download the three fine-tuned checkpoints, which are local experiment outputs.

```powershell
# Check an already restored local experiment; does not retrain.
.\.venv\Scripts\python.exe run.py verify --config configs/pilot.yaml
```

For a NEW training run, copy `configs/pilot.yaml`, change `study` to a NEW name, then run:

```powershell
.\.venv\Scripts\python.exe -u run.py all --config configs/my_new_study.yaml
```

Do not reuse `marc_en_zh_pilot` for a new run: completed models are skipped, and existing incomplete histories are protected against overwrite. Training is not bitwise guaranteed identical across hardware. `configs/full.yaml` is a proposal and has NOT been run.

## Publishing and rights

Read [upload instructions](docs/GITHUB_UPLOAD_ZH.md) and [third-party notices](THIRD_PARTY_NOTICES.md). No project-wide open-source licence was assigned without the owner's choice. Third-party materials retain their own terms. Do not assume public access grants redistribution permission.

```powershell
python scripts/check_publish.py
```

This checks the allowed publication surface; it does not certify legal clearance or guarantee detection of all secrets. Never force-add ignored data or weights. No remote repository was created or pushed by this packaging operation.
