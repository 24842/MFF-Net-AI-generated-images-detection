#!/usr/bin/env python3
# test_fusion.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
import datetime
from models import ExpertFusionTransformerDA

TEST_TITLE = "Chameleon Test"  

def metrics(y_true, y_score):
    y_pred = (y_score > 0.5).astype(int)
    ap = average_precision_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0]) if (y_true == 0).any() else 1.0
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1]) if (y_true == 1).any() else 1.0
    return ap, acc, r_acc, f_acc

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[3].item() != -1]
    if len(batch) == 0:
        return torch.empty(0, 640), torch.empty(0, 2048), torch.empty(0, 2048), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)


def test_all(model, semantic_features, lrhr_features, label_dict, device):
    sem_keys = set(semantic_features.keys())
    lrhr_keys = set(lrhr_features.keys())
    label_keys = set(label_dict.keys())
    common = list(sem_keys & lrhr_keys & label_keys)

    if not common:
        print("no common keys among semantic, lrhr, and label dictionaries.")
        return None
    
    semantic_subset = {k: semantic_features[k] for k in common}
    lrhr_subset = {k: lrhr_features[k] for k in common}
    sub_label = {k: label_dict[k] for k in common}

    class TestAllDataset(Dataset):
        def __init__(self, semantic_dict, lrhr_dict, label_dict):
            self.semantic_dict = semantic_dict
            self.lrhr_dict = lrhr_dict
            self.label_dict = label_dict
            self.keys = list(semantic_dict.keys())
        def __len__(self):
            return len(self.keys)
        def __getitem__(self, idx):
            key = self.keys[idx]
            try:
                sem = self.semantic_dict[key]
                lr = self.lrhr_dict[key]["lr"]
                hr = self.lrhr_dict[key]["hr"]
                label = self.label_dict[key]
                return sem, lr, hr, torch.tensor(label, dtype=torch.float32), torch.tensor(0, dtype=torch.long)
            except KeyError:
                return None

    sub_dataset = TestAllDataset(semantic_subset, lrhr_subset, sub_label)
    loader = DataLoader(sub_dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_fn)

    all_pred, all_true = [], []
    model.eval()
    with torch.no_grad():
        for sem, lr, hr, label, _ in tqdm(loader, desc='Overall Test', ncols=80):
            if len(sem) == 0:
                continue
            sem = sem.to(device).float()
            lr = lr.to(device).float()
            hr = hr.to(device).float()
            outputs = model(sem, lr, hr, domain_adapt=False)
            out = outputs[0]
            all_pred.append(torch.sigmoid(out).cpu())
            all_true.append(label)

    if len(all_pred) == 0:
        print("no valid samples in the test set.")
        return None

    all_pred = torch.cat(all_pred).numpy()
    all_true = torch.cat(all_true).numpy()
    n_real = (all_true == 0).sum()
    n_fake = (all_true == 1).sum()
    print(f"Overall Test: Real images {n_real}, Fake images {n_fake}")
    return metrics(all_true, all_pred)

 
# ---------------- 主入口 ----------------
def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    semantic_path = "/data1/zhanghongji/datasets/semantic_dinov3_Chameleon_test.pt"
    lrhr_path = "/data1/zhanghongji/datasets/lrhr_vgg_Cha_test.pt"
    label_path = "/data1/zhanghongji/datasets/label_Cha_te.pt"
    model_path = "/data1/zhanghongji/datasets/fusion_model_dinov3_progan_best.pt"

    label_dict = normalize_keys(torch.load(label_path, weights_only=False))
    semantic_features = normalize_keys(torch.load(semantic_path, weights_only=False))
    lrhr_features = normalize_keys(torch.load(lrhr_path, weights_only=False))


    try:
        first_key_sem = next(iter(semantic_features.keys()))
        semantic_dim = semantic_features[first_key_sem].shape[0]
        
        first_key_lrhr = next(iter(lrhr_features.keys()))
        lr_dim = lrhr_features[first_key_lrhr]["lr"].shape[0]
        hr_dim = lrhr_features[first_key_lrhr]["hr"].shape[0]
        
        print(f"Detected dimensions: semantic={semantic_dim}, lr={lr_dim}, hr={hr_dim}")
    except (StopIteration, KeyError):
        print("Unable to automatically detect dimensions, using default values.")
        semantic_dim, lr_dim, hr_dim = 1024, 2048, 2048

    model = ExpertFusionTransformerDA(
        sem_d=semantic_dim, lr_d=lr_dim, hr_d=hr_dim, n_dom=10
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    filtered_state_dict = {k: v for k, v in checkpoint.items() if "dom_clf" not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    print(f"Model loaded: {model_path}")

    # 5. Prepare log file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "/data1/zhanghongji/ImageDetection/logs/Chameleon_test_results.log")
    write_test_header(log_file)
    print(f"Log file: {log_file}")
    results = test_all(model, semantic_features, lrhr_features, label_dict, device)

    if results:
        ap, acc, r_acc, f_acc = results
        print("\n========== Final Results ==========")
        print(f"AP:{ap:.4f} | ACC:{acc:.4f} | R_ACC:{r_acc:.4f} | F_ACC:{f_acc:.4f}")
        
        # Append results to log
        append_log(log_file, "Overall", results)
        
        # Save as CSV
        save_path = "/data1/zhanghongji/ImageDetection/results/results_Chameleon.csv"
        df = pd.DataFrame([{"Model": "Chameleon", "AP": f"{ap:.4f}", "ACC": f"{acc:.4f}", "R_ACC": f"{r_acc:.4f}", "F_ACC": f"{f_acc:.4f}"}])
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")
    else:
        print("\nNo valid test results!")


def normalize_keys(d):
    """Normalize dictionary keys by replacing backslashes with forward slashes"""
    return {k.replace("\\", "/"): v for k, v in d.items()}


def append_log(log_path, model_name, metrics_tuple):
    """Append a single line of test results to the log file"""
    ap, acc, r_acc, f_acc = metrics_tuple
    ts = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    line = f"{ts} | {model_name} | AP:{ap:.4f} | ACC:{acc:.4f} | R_ACC:{r_acc:.4f} | F_ACC:{f_acc:.4f}\n"
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    except Exception:
        pass
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def write_test_header(log_path):
    """Write test title and timestamp to the log file"""
    ts = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    header_lines = [
        "=" * 80,
        f"Test Title: {TEST_TITLE}",
        f"Test Time: {ts}",
        "=" * 80,
        ""
    ]
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    except Exception:
        pass
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(header_lines))


if __name__ == "__main__":
    main()