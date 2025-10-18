import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    # read the input images
    @staticmethod
    def _load_input_image(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    # read the mask images
    @staticmethod
    def _load_target_image(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
            
    def __init__(self, input_root, target_root, transform_input=None,
                 transform_target=None, seed_fn=None):   
        self.input_root = input_root
        self.target_root = target_root
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.seed_fn = seed_fn
        # sort the ids     
        self.input_ids = sorted(img for img in os.listdir(self.input_root))
        self.target_ids = sorted(img for img in os.listdir(self.target_root))
        assert(len(self.input_ids) == len(self.target_ids))

    # set random number seed
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if self.seed_fn:
            self.seed_fn(seed)
        
    def __getitem__(self, idx):
        input_img = self._load_input_image(
            os.path.join(self.input_root, self.input_ids[idx]))
        target_img = self._load_target_image(
            os.path.join(self.target_root, self.target_ids[idx]))
        
        if self.transform_input:
            # ensure that the input and output have the same randomness.
            seed = random.randint(0, 2**32)
            self._set_seed(seed)
            input_img = self.transform_input(input_img)
            self._set_seed(seed)
            target_img = self.transform_target(target_img)  
        return input_img, target_img, self.input_ids[idx]
        
    def __len__(self):
        return len(self.input_ids)