import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save log files')
        self.parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        self.parser.add_argument('--device', type=str, default='cuda', help='device to use for training / testing')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        if self.opt.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU.")
            self.opt.device = 'cpu'
            
        os.makedirs(self.opt.log_dir, exist_ok=True)
        
        print("="*20 + " Options " + "="*20)
        for k, v in sorted(vars(self.opt).items()):
            print(f"{k}: {v}")
        print("="*49)
        
        return self.opt