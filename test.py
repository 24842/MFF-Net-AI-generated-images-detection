import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime 

from models import ExpertFusionTransformerDA
from data.dataset import FusionDataset
from utils import calculate_metrics
from utils import setup_logger 
from options.test_options import TestOptions

def get_subset_name(path, split_folder='test'):
    path = str(path).replace('\\', '/')
    parts = path.split('/')
    parts = [p for p in parts if p and p != '.']

    possible_anchors = ['GenImage', 'test', 'validation', 'val']
    if split_folder and split_folder != 'test': 
        possible_anchors.insert(0, split_folder)

    for anchor in possible_anchors:
        if anchor in parts:
            try:
                idx = parts.index(anchor)
                if idx + 1 < len(parts):
                    return parts[idx + 1]
            except:
                continue
    if len(parts) > 0:
        return parts[0]
        
    return "Unknown"

def main():
    opt = TestOptions().parse()
    device = opt.device

    os.makedirs(opt.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"test_results_{timestamp}.log"

    logger = setup_logger(opt.log_dir, filename=log_filename)
    logger.info("="*60)
    logger.info(f"Test Started at {timestamp}")
    logger.info(f"Model Path: {opt.model_path}")
    logger.info("="*60)
    
    logger.info(f"Loading label file: {opt.label_test}")
    raw_label_dict = torch.load(opt.label_test, weights_only=False)
    raw_label_dict = {k.replace("\\", "/"): v for k, v in raw_label_dict.items()}
    
    dataset = FusionDataset(opt.sem_test, opt.lrhr_test, raw_label_dict)
    
    valid_keys = dataset.keys 
    
    logger.info(f"[Sanity Check] Dataset internal keys: {len(valid_keys)}")
    logger.info(f"[Sanity Check] First key example: {valid_keys[0]}")
    logger.info(f"[Sanity Check] Detected Subset: {get_subset_name(valid_keys[0], opt.split_folder)}")

    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    
    model = ExpertFusionTransformerDA(
        sem_d=dataset.sem_dim, lr_d=dataset.lr_dim, hr_d=dataset.hr_dim, n_dom=10
    ).to(device)
    
    logger.info(f"Loading checkpoint from {opt.model_path}...") 
    ckpt = torch.load(opt.model_path, map_location=device, weights_only=False)
    state_dict = {k: v for k, v in ckpt.items() if 'dom_clf' not in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    all_pred, all_true = [], []
    
    current_idx = 0
    
    with torch.no_grad():
        for sem, lr, hr, label, _ in tqdm(loader, desc="Testing"):
            batch_size = sem.size(0)
            
            sem, lr, hr = sem.to(device), lr.to(device), hr.to(device)
            out, _, sem_o, lr_o, hr_o = model(sem, lr, hr, domain_adapt=False)
            
            fusion_p = torch.sigmoid(out).squeeze()
            sem_p = torch.sigmoid(sem_o).squeeze()
            lr_p  = torch.sigmoid(lr_o).squeeze()
            hr_p  = torch.sigmoid(hr_o).squeeze()
            
            final_p = fusion_p.clone()
            conf_thresh = 0.9
            mask_sem = (sem_p > conf_thresh) | (sem_p < 1-conf_thresh)
            mask_lr = (lr_p > conf_thresh) | (lr_p < 1-conf_thresh)
            mask_hr = (hr_p > conf_thresh) | (hr_p < 1-conf_thresh)
            
            final_p[mask_sem] = sem_p[mask_sem]
            final_p[~mask_sem & mask_lr] = lr_p[~mask_sem & mask_lr]
            final_p[~mask_sem & ~mask_lr & mask_hr] = hr_p[~mask_sem & ~mask_lr & mask_hr]
            
            all_pred.append(final_p.cpu())
            all_true.append(label)
            
            current_idx += batch_size
            
    preds = torch.cat(all_pred).numpy()
    gts = torch.cat(all_true).numpy()
    
    if len(preds) != len(valid_keys):
        logger.error(f"FATAL ERROR: Preds ({len(preds)}) != Keys ({len(valid_keys)})")
        return

    df_results = pd.DataFrame({
        'key': valid_keys,  
        'pred': preds,
        'gt': gts
    })
    
    df_results['subset'] = df_results['key'].apply(lambda x: get_subset_name(x, opt.split_folder))
    
    final_metrics_list = []
    
    logger.info("\n" + "="*80)

    unique_subsets = sorted(df_results['subset'].unique())
    
    for subset_name in unique_subsets:
        sub_df = df_results[df_results['subset'] == subset_name]
        if len(sub_df) == 0: continue
            
        s_ap, s_acc, s_r_acc, s_f_acc, s_auc = calculate_metrics(sub_df['gt'].values, sub_df['pred'].values)
        
        final_metrics_list.append({
            "Subset": subset_name,
            "Count": len(sub_df),
            "AP": s_ap, "ACC": s_acc, "R_ACC": s_r_acc, "F_ACC": s_f_acc, "AUC": s_auc
        })
        
        logger.info(f"| {subset_name:<22} | AP:{s_ap:.6f} | ACC:{s_acc:.6f} | R_ACC:{s_r_acc:.6f} | F_ACC:{s_f_acc:.6f} | AUC:{s_auc:.6f}")

    logger.info("="*80)

    total_ap, total_acc, total_r_acc, total_f_acc, total_auc = calculate_metrics(gts, preds)
    
    final_metrics_list.append({
        "Subset": "Overall",
        "Count": len(df_results),
        "AP": total_ap, "ACC": total_acc, 
        "R_ACC": total_r_acc, "F_ACC": total_f_acc, 
        "AUC": total_auc
    })
    
    logger.info(f"| {'Overall':<22} | AP:{total_ap:.6f} | ACC:{total_acc:.6f} | R_ACC:{total_r_acc:.6f} | F_ACC:{total_f_acc:.6f} | AUC:{total_auc:.6f}")
    logger.info("="*80)
    
    df_out = pd.DataFrame(final_metrics_list)
    cols = ["Subset", "Count", "AP", "ACC", "R_ACC", "F_ACC", "AUC"]
    df_out = df_out[cols]
    
    df_out.to_csv(opt.save_csv, index=False)
    logger.info(f"Detailed results saved to csv: {opt.save_csv}")

if __name__ == "__main__":
    main()