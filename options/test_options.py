from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--sem_test', type=str, required=True, help='Path to test semantic features')
        self.parser.add_argument('--lrhr_test', type=str, required=True, help='Path to test LR/HR features')
        self.parser.add_argument('--label_test', type=str, required=True, help='Path to test labels')
        self.parser.add_argument('--model_path', type=str, required=True, help='Path to loaded model checkpoint')
        self.parser.add_argument('--save_csv', type=str, default='results.csv', help='Path to save results csv')
        self.parser.add_argument('--split_folder', type=str, default='test', help='Folder name for subset splitting')
        self.parser.set_defaults(batch_size=64)