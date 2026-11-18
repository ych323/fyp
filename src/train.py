from __future__ import annotations

import math
import platform
import time

import numpy as np
import psutil
from safetensors.torch import save_file
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .common import ROOT, experiment_dir, fingerprint, model_path, processed_dir, save_json, seed_all
from .data import EncodedReviews
from .metrics import classification_metrics
from .model import SentimentModel


@torch.inference_mode()
def predict(model, dataset, config, device):
    model.eval()
    logits, labels = [], []
    loader = DataLoader(dataset, batch_size=config["eval_batch_size"], shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    for batch in loader:
        y = batch.pop("labels")
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs, _ = model(**inputs)
        logits.append(outputs.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def train(config, name):
    seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this run. Activate the project CUDA environment.")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    base = "mbert" if name == "mbert" else "xlmr"
    dann = name == "xlmr_dann"
    out = experiment_dir(config) / name
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "models" / "finetuned" / config["study"] / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    training = EncodedReviews(config, base, "en_train")
    selection = EncodedReviews(config, base, "en_selection")
    target = EncodedReviews(config, base, "zh_unlabeled") if dann else None
    generator = torch.Generator().manual_seed(config["seed"])
    source_loader = DataLoader(training, batch_size=config["train_batch_size"], shuffle=True,
                               generator=generator, num_workers=0, pin_memory=True)
    target_loader = DataLoader(target, batch_size=config["train_batch_size"], shuffle=True,
                               generator=torch.Generator().manual_seed(config["seed"] + 1),
                               num_workers=0, pin_memory=True) if dann else None
    # Reseed after preprocessing so initial task head is identical for XLM-R and XLM-R+DANN.
    seed_all(config["seed"])
    model = SentimentModel(model_path(base), config["num_labels"], dann=dann).to(device)
    if config["gradient_checkpointing"]:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], foreach=False)
    accumulation = config["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(source_loader) / accumulation)
    total_steps = steps_per_epoch * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["warmup_ratio"]), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss()
    history = out / "training_history.jsonl"
    if history.exists():
        raise RuntimeError("Training log already exists. Use a new study name to retain previous runs: " + str(history))
    code = {str(p.relative_to(ROOT)): fingerprint(p) for p in (ROOT / "src").glob("*.py")}
    metadata = {"status": "training", "model": name, "base": base, "config": config,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "gpu": torch.cuda.get_device_name(0), "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "torch": torch.__version__, "python": platform.python_version(),
                "system_ram_gb": psutil.virtual_memory().total / 1e9,
                "source_data_sha256": fingerprint(processed_dir(config) / "en_train.jsonl"),
                "target_data_sha256": fingerprint(processed_dir(config) / "zh_unlabeled.jsonl") if dann else None,
                "code_sha256": code, "best_selection_mae": None, "epochs": []}
    save_json(out / "run_metadata.json", metadata)
    best = float("inf")
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    for epoch in range(config["epochs"]):
        model.train()
        target_iterator = iter(target_loader) if dann else None
        running = {"task_loss": 0.0, "domain_loss": 0.0, "domain_correct": 0, "domain_count": 0, "batches": 0}
        for step, batch in enumerate(source_loader):
            labels = batch.pop("labels").to(device)
            source_inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            progress = (epoch * len(source_loader) + step) / max(1, config["epochs"] * len(source_loader) - 1)
            strength = config["dann_max_lambda"] * (2 / (1 + math.exp(-10 * progress)) - 1) if dann else 0.0
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(source_loader) - group_start)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if dann:
                    try:
                        target_batch = next(target_iterator)
                    except StopIteration:
                        target_iterator = iter(target_loader)
                        target_batch = next(target_iterator)
                    assert "labels" not in target_batch
                    combined = {k: torch.cat([v, target_batch[k].to(device, non_blocking=True)]) for k, v in source_inputs.items()}
                    logits, domain_logits = model(strength=strength, **combined)
                    task_loss = loss_fn(logits[:len(labels)], labels)
                    domain_labels = torch.cat([torch.zeros(len(labels), dtype=torch.long, device=device),
                                               torch.ones(len(target_batch["input_ids"]), dtype=torch.long, device=device)])
                    domain_loss = loss_fn(domain_logits, domain_labels)
                    loss = (task_loss + domain_loss) / group_size
                    running["domain_correct"] += int((domain_logits.detach().argmax(1) == domain_labels).sum())
                    running["domain_count"] += len(domain_labels)
                else:
                    logits, _ = model(**source_inputs)
                    task_loss = loss_fn(logits, labels)
                    domain_loss = task_loss.new_zeros(())
                    loss = task_loss / group_size
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; inspect configuration.")
            scaler.scale(loss).backward()
            running["task_loss"] += float(task_loss.detach())
            running["domain_loss"] += float(domain_loss.detach())
            running["batches"] += 1
            if (step + 1) % accumulation == 0 or step + 1 == len(source_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if updates % 25 == 0 or step + 1 == len(source_loader):
                    elapsed = time.perf_counter() - start
                    item = {"epoch": epoch + 1, "step": updates, "total_steps": total_steps,
                            "task_loss": running["task_loss"] / running["batches"],
                            "domain_loss": running["domain_loss"] / running["batches"],
                            "domain_train_accuracy": running["domain_correct"] / max(1, running["domain_count"]),
                            "grl_strength": strength, "learning_rate": scheduler.get_last_lr()[0],
                            "elapsed_seconds": elapsed, "eta_seconds": elapsed / updates * (total_steps - updates),
                            "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9}
                    with open(history, "a", encoding="utf-8") as f:
                        import json
                        f.write(json.dumps(item) + "\n")
                    print(name + " " + json.dumps(item), flush=True)
        logits, labels = predict(model, selection, config, device)
        metrics = classification_metrics(logits, labels, bins=config["ece_bins"])
        metadata["epochs"].append({"epoch": epoch + 1, "selection": metrics})
        if metrics["mae"] < best:
            best = metrics["mae"]
            state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
            save_file(state, str(checkpoint / "best.safetensors"))
            del state
            save_json(checkpoint / "architecture.json", {"base": base, "dann": dann, "num_labels": 5})
            metadata["best_epoch"] = epoch + 1
            metadata["best_selection_mae"] = best
        print(f"{name} epoch {epoch + 1} selection MAE={metrics['mae']:.4f}, accuracy={metrics['accuracy']:.4f}", flush=True)
        save_json(out / "run_metadata.json", metadata)
    metadata.update(status="trained", training_seconds=time.perf_counter() - start,
                    peak_cuda_gb=torch.cuda.max_memory_allocated() / 1e9,
                    checkpoint_sha256=fingerprint(checkpoint / "best.safetensors"))
    save_json(out / "run_metadata.json", metadata)
    print(f"Completed {name}; best epoch={metadata['best_epoch']}; {metadata['training_seconds'] / 60:.1f} minutes", flush=True)
