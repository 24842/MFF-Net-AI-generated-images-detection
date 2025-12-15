import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import datetime

from data.dataset import FusionDataset
from models import ExpertFusionTransformerDA
from utils import FocalLoss
from utils.logger import setup_logger 
from options.train_options import TrainOptions

def normalize_keys(d):
    return {k.replace("\\", "/"): v for k, v in d.items()}

def main():
    opt = TrainOptions().parse()
    
    os.makedirs(opt.save_dir, exist_ok=True)
    
    device = torch.device(opt.device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = setup_logger(opt.log_dir, filename=f"train_{timestamp}.log")
    
    logger.info("="*40)
    logger.info(f"Training Started at {timestamp}")
    logger.info(f"Args: {opt}")
    logger.info("="*40)
    
    logger.info("Loading labels...")
    label_train = normalize_keys(torch.load(opt.label_train, weights_only=False))
    label_val = normalize_keys(torch.load(opt.label_val, weights_only=False))

    labels_list = list(label_train.values())
    pos_count = sum(labels_list)
    neg_count = len(labels_list) - pos_count
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Pos samples: {pos_count}, Neg samples: {neg_count}, Pos Weight: {pos_weight:.2f}")

    train_set = FusionDataset(opt.sem_train, opt.lrhr_train, label_train)
    val_set = FusionDataset(opt.sem_val, opt.lrhr_val, label_val)
    
    logger.info(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    model = ExpertFusionTransformerDA(
        sem_d=train_set.sem_dim, 
        lr_d=train_set.lr_dim,
        hr_d=train_set.hr_dim, 
        n_dom=len(train_set.domains)
    ).to(device)

    opt_optim = optim.Adam([
        {'params': model.sem_proj.parameters(), 'lr': opt.lr},
        {'params': model.lr_proj.parameters(),  'lr': opt.lr},
        {'params': model.hr_proj.parameters(),  'lr': opt.lr},
        {'params': model.gate.parameters(),      'lr': opt.lr},
        {'params': model.fuse.parameters(),      'lr': opt.lr},
        {'params': model.clf.parameters(),       'lr': opt.lr * 10},     
        {'params': model.dom_clf.parameters(),   'lr': opt.lr * 10},
        {'params': model.sem_cls.parameters(),   'lr': opt.lr * 10},
        {'params': model.lr_cls.parameters(),    'lr': opt.lr * 10},
        {'params': model.hr_cls.parameters(),    'lr': opt.lr * 10},
    ])

    sched = optim.lr_scheduler.ReduceLROnPlateau(opt_optim, mode='max', factor=0.7, patience=5, verbose=True)
    crit = FocalLoss(alpha=0.75, gamma=2.0)
    dom_crit = torch.nn.CrossEntropyLoss()

    best_auc = 0.0
    stale = 0  
    
    logger.info("Start Training Loop...")
    
    for epoch in range(opt.epochs):
        model.train()
        losses = {'cls': 0., 'dom': 0., 'con': 0.}
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, ncols=100, desc=f'Epoch {epoch+1}/{opt.epochs}')
        for idx, (sem, lr, hr, label, dom) in enumerate(pbar):
            sem, lr, hr, label, dom = sem.to(device), lr.to(device), hr.to(device), label.to(device), dom.to(device)
            
            p = float(idx + epoch*len(train_loader)) / (opt.epochs*len(train_loader))
            alpha = 2. / (1. + np.exp(-10*p)) - 1

            opt_optim.zero_grad()
            out, dom_out, sem_o, lr_o, hr_o = model(sem, lr, hr, domain_adapt=True, alpha=alpha)

            cls_loss = crit(out.squeeze(), label)
            dom_loss = dom_crit(dom_out, dom)
            
            sem_p = torch.sigmoid(sem_o).squeeze()
            lr_p  = torch.sigmoid(lr_o).squeeze()
            hr_p  = torch.sigmoid(hr_o).squeeze()
            fusion_sig = torch.sigmoid(out).squeeze()
            con_loss = F.mse_loss(fusion_sig, (sem_p+lr_p+hr_p)/3)
            
            loss = cls_loss + 0.1*dom_loss + 0.2*con_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_optim.step()

            losses['cls'] += cls_loss.item()
            losses['dom'] += dom_loss.item()
            losses['con'] += con_loss.item()
            
            preds = (fusion_sig > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)
            
            if idx % 10 == 0:
                pbar.set_postfix({
                    'cls': cls_loss.item(), 
                    'acc': correct/total
                })

        avg_cls_loss = losses['cls'] / len(train_loader)
        avg_train_acc = correct / total
        logger.info(f"Epoch {epoch+1} Train | Cls Loss: {avg_cls_loss:.4f} | Acc: {avg_train_acc:.4f}")

        model.eval()
        preds_list, gts_list = [], []
        with torch.no_grad():
            for sem, lr, hr, label, _ in tqdm(val_loader, desc='Validation', ncols=80):
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
                
                final_p[mask_hr] = hr_p[mask_hr]
                final_p[mask_lr] = lr_p[mask_lr]
                final_p[mask_sem] = sem_p[mask_sem]
                
                preds_list.append(final_p.cpu())
                gts_list.append(label)

        if len(preds_list) > 0:
            preds_arr = torch.cat(preds_list).numpy()
            gts_arr = torch.cat(gts_list).numpy()
            val_auc = roc_auc_score(gts_arr, preds_arr)
        else:
            val_auc = 0.0

        logger.info(f'Epoch {epoch+1} Val   | AUC: {val_auc:.4f} (Best: {best_auc:.4f})')
        sched.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            stale = 0
            save_path = os.path.join(opt.save_dir, 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            logger.info(f">>> New Best Model Saved to {save_path}")
        else:
            stale += 1
            logger.info(f"No improvement for {stale} epochs. (Patience: {opt.patience})")
            if stale >= opt.patience:
                logger.info(f">>> Early stopping triggered at epoch {epoch+1}. Best AUC: {best_auc:.4f}")
                break
    
    torch.save(model.state_dict(), os.path.join(opt.save_dir, 'final_model.pt'))
    logger.info("Training Finished.")

if __name__ == '__main__':
    main()