from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--sem_train', type=str, required=True, help='Path to training semantic features')
        self.parser.add_argument('--lrhr_train', type=str, required=True, help='Path to training LR/HR features')
        self.parser.add_argument('--label_train', type=str, required=True, help='Path to training labels')
        self.parser.add_argument('--sem_val', type=str, required=True, help='Path to val semantic features')
        self.parser.add_argument('--lrhr_val', type=str, required=True, help='Path to val LR/HR features')
        self.parser.add_argument('--label_val', type=str, required=True, help='Path to val labels')
        
        self.parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
        self.parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.parser.add_argument('--patience', type=int, default=6, help='Early stopping patience')