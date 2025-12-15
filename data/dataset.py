import torch
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, semantic_path, lrhr_path, label_dict):
        self.sem = torch.load(semantic_path, weights_only=False, map_location='cpu')
        self.lrhr = torch.load(lrhr_path, weights_only=False, map_location='cpu')
        self.label_dict = label_dict
        
        self.keys = sorted(list(set(self.sem.keys()) & set(self.lrhr.keys()) & set(self.label_dict.keys())))
        
        self.domain_labels = {}
        for k in self.keys:
            dom = k.split('/')[0] if '/' in k else 'unknown'
            self.domain_labels[k] = dom
            
        self.domains = sorted(list(set(self.domain_labels.values())))
        self.dom2id = {d: i for i, d in enumerate(self.domains)}
        if self.keys:
            k0 = self.keys[0]
            self.sem_dim = self.sem[k0].shape[0]
            self.lr_dim = self.lrhr[k0]['lr'].shape[0]
            self.hr_dim = self.lrhr[k0]['hr'].shape[0]
        else:
            self.sem_dim, self.lr_dim, self.hr_dim = 640, 2048, 2048

        print(f'[Dataset] Loaded {len(self.keys)} samples from {len(self.domains)} domains.')

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        return (self.sem[k],
                self.lrhr[k]['lr'],
                self.lrhr[k]['hr'],
                torch.tensor(self.label_dict[k], dtype=torch.float32),
                torch.tensor(self.dom2id[self.domain_labels[k]], dtype=torch.long))