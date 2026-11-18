from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, experiment_dir, save_json

NAMES = {"mbert": "mBERT", "xlmr": "XLM-R Base", "xlmr_dann": "XLM-R + DANN"}
METRICS = ["accuracy", "macro_f1", "mae", "ece", "nll", "brier"]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(rows):
    if not rows:
        return "<p>Pending / 等待实验完成</p>"
    keys = list(rows[0])
    parts = ["<table><thead><tr>" + "".join("<th>" + html.escape(k) + "</th>" for k in keys) + "</tr></thead><tbody>"]
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        parts.append("<tr>" + "".join("<td>" + html.escape(fmt(r[k])) + "</td>" for k in keys) + "</tr>")
    return "".join(parts) + "</tbody></table>"


def report(config):
    study = experiment_dir(config)
    out = ROOT / "reports" / config["study"]
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    runs = {}
    for name in NAMES:
        p = study / name / "metrics.json"
        if p.exists():
            runs[name] = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for name, run in runs.items():
        for split, values in run["splits"].items():
            for stage in ("before", "after"):
                m = values[stage]
                rows.append({"model": name, "test_split": split, "calibration": stage,
                             "target_unlabeled_used": name == "xlmr_dann", "n": m["n"],
                             **{k: m[k] for k in METRICS}, "temperature": m["temperature"]})
    write_csv(out / "all_results.csv", rows)
    p1 = [r for r in rows if r["model"] == "mbert" and r["calibration"] == "before"]
    p2 = [r for r in rows if r["model"] in ("mbert", "xlmr") and r["calibration"] == "before"]
    p3 = [r for r in rows if r["model"] in ("xlmr", "xlmr_dann") and r["calibration"] == "before"]
    p6 = rows
    for paper, subset in [(1, p1), (2, p2), (3, p3), (6, p6)]:
        write_csv(out / f"paper{paper}_results.csv", subset)
    plot_figures(runs, study, figures)
    resources = []
    for name in runs:
        meta = json.loads((study / name / "run_metadata.json").read_text(encoding="utf-8"))
        resources.append({"model": name, "parameters": meta["parameters"],
                          "training_minutes": meta["training_seconds"] / 60,
                          "peak_cuda_GB": meta["peak_cuda_gb"],
                          "best_epoch": meta["best_epoch"], "selection_MAE": meta["best_selection_mae"]})
    write_csv(out / "resources.csv", resources)
    original = [
        {"setting": "English supervised, full 200k, 15 epochs", "original_accuracy": 0.633, "original_MAE": 0.430},
        {"setting": "English to Chinese zero-shot, full 200k, 15 epochs", "original_accuracy": 0.397, "original_MAE": 0.932},
    ]
    comparisons = []
    if "xlmr" in runs and "xlmr_dann" in runs:
        a = runs["xlmr"]["splits"]["zh_test"]["before"]
        b = runs["xlmr_dann"]["splits"]["zh_test"]["before"]
        comparisons.append(f"DANN在中文测试集的准确率变化为 {(b['accuracy'] - a['accuracy']) * 100:+.2f} 个百分点，MAE变化为 {b['mae'] - a['mae']:+.4f}。这是单种子初步结果。")
    if "mbert" in runs and "xlmr" in runs:
        a = runs["mbert"]["splits"]["zh_test"]["before"]
        b = runs["xlmr"]["splits"]["zh_test"]["before"]
        comparisons.append(f"XLM-R相对mBERT的中文准确率变化为 {(b['accuracy'] - a['accuracy']) * 100:+.2f} 个百分点。")
    for name, run in runs.items():
        a, b = (run["splits"]["zh_test"][s] for s in ("before", "after"))
        comparisons.append(f"{NAMES[name]}的中文ECE：{a['ece']:.4f} → {b['ece']:.4f}；温度参数仅用英文校准集拟合。")
    status = f"Completed models: {len(runs)}/3 / 已完成模型：{len(runs)}/3"
    description = (
        f"单次初步实验 / single-seed pilot · seed={config['seed']} · English training n={config['train_samples']:,} · "
        f"epochs={config['epochs']} · max length={config['max_length']} · RTX 4060 Laptop 8 GB")
    sections = [
        ("论文1 · MARC / mBERT benchmark", "复现其五星分类和English→Chinese设置。缩小训练样本、训练轮数和输入长度；本次是pilot，原论文数字仅供背景参考。", p1),
        ("论文2 · XLM-R downstream experiment", "使用公开预训练权重进行MARC下游微调。MARC不是XLM-R原论文的评测任务，因此这里是下游应用实验，不能声称复制原论文结果表。", p2),
        ("论文3 · DANN method adaptation", "固定相同英文样本，对比source-only与使用无标签中文数据的DANN。域标签为语言；使用Transformer和五星任务，区别于原论文词袋二分类。", p3),
        ("论文6 · Temperature Scaling", "每个模型使用2,500条英文校准样本拟合一个T；英文/中文测试集只用于最终评估。中文校准效果不保证改善。", p6),
    ]
    body = f"<h1>MARC English → Chinese</h1><p class='subtitle'>初步复现报告 / Pilot Reproduction Report</p><p>{html.escape(description)}</p><p class='badge'>{status}</p>"
    body += "<h2>结果解读 / Findings</h2>" + "".join("<p>" + html.escape(x) + "</p>" for x in comparisons)
    body += "<p>所有数字由保存的预测结果自动计算。一次随机种子不能证明稳定提升，也不构成统计显著性结论。</p>"
    body += "<h2>实验协议 / Protocol</h2>" + table([
        {"split": "English training", "n": config["train_samples"], "purpose": "same stratified sample for all three models"},
        {"split": "Chinese training", "n": config["target_unlabeled_samples"], "purpose": "DANN only, uniform sample without inspecting labels; labels removed"},
        {"split": "English selection", "n": 2500, "purpose": "select checkpoint with minimum MAE"},
        {"split": "English calibration", "n": 2500, "purpose": "fit temperature after model selection"},
        {"split": "English test", "n": 5000, "purpose": "final evaluation only"},
        {"split": "Chinese test", "n": 5000, "purpose": "final evaluation only"},
    ])
    dataset_overview = ROOT / "data/provenance/dataset_overview.json"
    if dataset_overview.exists():
        overview = json.loads(dataset_overview.read_text(encoding="utf-8"))
        body += "<h2>数据分布 / Dataset distribution</h2>" + table([
            {"language": lang, "training_rows": overview[f"{lang}_train"]["n"],
             "mean_body_characters": overview[f"{lang}_train"]["mean_body_characters"],
             "books_fraction": overview[f"{lang}_train"]["books_fraction"]} for lang in ("en", "zh")])
        body += "<p>这是完整原始训练集的描述统计。中英文产品类别分布差异明显，因此本次迁移包含语言及类别分布共同变化。</p>"
    for title, note, subset in sections:
        body += "<h2>" + title + "</h2><p>" + note + "</p>" + table(subset)
    body += "<h2>MARC原论文参考数值 / Original-paper context</h2><p>来自Table 2和Table 3，body + title + category。原论文200k训练样本、15轮、8e-7学习率、body截断180wordpieces；本次10k、2轮、2e-5、总长128。禁止据此宣称超过原论文。</p>" + table(original)
    body += "<h2>图表 / Figures</h2>"
    for fn, caption in [("performance.png", "Accuracy and ordinal MAE"), ("reliability.png", "Reliability diagrams: calibration fitted on English only"),
                        ("training.png", "Training losses and English selection MAE"), ("confusion.png", "Confusion matrices before calibration")]:
        if (figures / fn).exists():
            body += f"<figure><img src='figures/{fn}' alt='{caption}'><figcaption>{caption}</figcaption></figure>"
    body += "<h2>资源记录 / Resources</h2>" + table(resources)
    body += "<h2>研究边界 / Limitations</h2><p>单种子、小样本、固定超参数，尚未做显著性检验或多种子重复。目标语言类别分布不同，且product category为英文输入字段；结果包含语言和类别分布共同变化，不能归因于纯语言效应。DANN看到目标语言无标签文本，因此与zero-shot模型的数据访问权限不同。训练域分类准确率接近50%也不能单独证明有效对齐。</p>"
    body += "<p>Source-only temperature calibration may fail under target-language distribution shift. ECE uses 15 equal-width bins. Brier is the mean per-example sum of squared class-probability errors, not divided by five. Accuracy and macro-F1 are proportions (0–1); MAE is measured in star-rating units.</p>"
    body += "<h2>下载与审计 / Downloads and audit</h2><ul>"
    for fn in ["all_results.csv", "paper1_results.csv", "paper2_results.csv", "paper3_results.csv", "paper6_results.csv", "resources.csv"]:
        if (out / fn).exists():
            body += f"<li><a href='{fn}'>{fn}</a></li>"
    body += "<li><a href='../../docs/PROTOCOL.md'>实验协议 / Protocol</a></li><li><a href='../../data/provenance/downloads.json'>版本与SHA256 / Download manifest</a></li><li><a href='../../references/REFERENCES.md'>论文链接 / References</a></li></ul>"
    css = "body{font:16px/1.65 'Segoe UI','Microsoft YaHei',sans-serif;color:#183047;background:#f7f9fc;margin:0;padding:40px 5vw;max-width:1500px}h1{font-size:38px;margin-bottom:0}h2{margin-top:38px;border-bottom:2px solid #d7e3ef;padding-bottom:8px}.subtitle{font-size:22px;color:#496b8f}.badge{background:#e1edfa;display:inline-block;padding:8px 16px;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px;background:white;display:block;overflow:auto}th,td{padding:10px 12px;border:1px solid #d8e1eb;text-align:left;white-space:nowrap}th{background:#183f64;color:white}tr:nth-child(even){background:#f0f5fa}img{width:100%;height:auto}figure{margin:24px 0;background:white;padding:15px}a{color:#1367a8}@media print{body{padding:10px;font-size:11px}h2,figure{break-inside:avoid}table{font-size:9px}}"
    document = "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>MARC Pilot Reproduction</title><style>" + css + "</style></head><body>" + body + "</body></html>"
    (out / "index.html").write_text(document, encoding="utf-8")
    save_json(out / "report_manifest.json", {"models_completed": list(runs), "config": config, "status": "complete" if len(runs) == 3 else "partial"})
    print("Report: " + str(out / "index.html"), flush=True)


def plot_figures(runs, study, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = list(runs)
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(names))
    for i, (split, color) in enumerate([("en_test", "#3279b7"), ("zh_test", "#eea348")]):
        for ax, metric, title in zip(axes, ["accuracy", "mae"], ["Accuracy (higher is better)", "Ordinal MAE (lower is better)"]):
            vals = [runs[n]["splits"][split]["before"][metric] for n in names]
            bars = ax.bar(x + (i - 0.5) * .34, vals, width=.34, label=split, color=color)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            ax.set_xticks(x, [NAMES[n] for n in names], rotation=12)
            ax.set_title(title)
            overall_max = max(runs[n]["splits"][s]["before"][metric] for n in names for s in ("en_test", "zh_test"))
            ax.set_ylim(0, max(0.1, overall_max) * 1.25 if metric == "mae" else 1)
            ax.legend()
    fig.suptitle("English-to-Chinese MARC: single-seed pilot")
    fig.savefig(out / "performance.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 8), squeeze=False, constrained_layout=True)
    for c, name in enumerate(names):
        for r, split in enumerate(("en_test", "zh_test")):
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
            for stage, color in [("before", "#c85647"), ("after", "#3279b7")]:
                metrics = runs[name]["splits"][split][stage]
                bins = [b for b in metrics["reliability_bins"] if b["count"]]
                ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color=color,
                        label=f"{stage} ECE={metrics['ece']:.3f}")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy",
                   title=f"{NAMES[name]} | {split}")
            ax.legend(fontsize=8)
    fig.savefig(out / "reliability.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name in names:
        log = study / name / "training_history.jsonl"
        history = [json.loads(s) for s in log.read_text(encoding="utf-8").splitlines()]
        axes[0].plot([h["step"] for h in history], [h["task_loss"] for h in history], label=NAMES[name])
        meta = json.loads((study / name / "run_metadata.json").read_text(encoding="utf-8"))
        axes[1].plot([h["epoch"] for h in meta["epochs"]], [h["selection"]["mae"] for h in meta["epochs"]], "o-", label=NAMES[name])
    axes[0].set(xlabel="Optimizer step", ylabel="Mean task loss within epoch", title="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="MAE", title="English selection set (2,500 examples)")
    for ax in axes:
        ax.legend()
    fig.savefig(out / "training.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, len(names), figsize=(4.5 * len(names), 8), squeeze=False, constrained_layout=True)
    for c, name in enumerate(names):
        for r, split in enumerate(("en_test", "zh_test")):
            ax = axes[r, c]
            matrix = np.array(runs[name]["splits"][split]["before"]["confusion_matrix"])
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1000)
            for y in range(5):
                for x in range(5):
                    ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="white" if matrix[y, x] > 500 else "black", fontsize=9)
            ax.set(xticks=range(5), yticks=range(5), xticklabels=range(1, 6), yticklabels=range(1, 6),
                   xlabel="Predicted stars", ylabel="True stars", title=f"{NAMES[name]} | {split}")
    fig.savefig(out / "confusion.png", dpi=180)
    plt.close(fig)
