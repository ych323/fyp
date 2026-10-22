"""Describe the retained original data using streaming reads."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
overview = {}
table = []
for language in ("en", "zh"):
    for split in ("train", "validation", "test"):
        path = ROOT / "data/raw/marc" / language / (split + ".jsonl.gz")
        stars, categories, lengths = Counter(), Counter(), []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                stars[int(r["stars"])] += 1
                categories[r["product_category"]] += 1
                lengths.append(len(r["review_body"]))
        row = {"language": language, "split": split, "n": len(lengths),
               "compressed_bytes": path.stat().st_size,
               "mean_body_characters": statistics.mean(lengths),
               "median_body_characters": statistics.median(lengths),
               "books_fraction": categories["book"] / len(lengths)}
        overview[f"{language}_{split}"] = {**row, "stars": dict(stars), "categories": dict(categories.most_common())}
        table.append(row)
out = ROOT / "data/provenance"
out.mkdir(exist_ok=True)
(out / "dataset_overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
with open(out / "dataset_overview.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
print(json.dumps(table, indent=2), flush=True)
