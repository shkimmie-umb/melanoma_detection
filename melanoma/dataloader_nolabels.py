import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pathlib

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        # Filter image files
        self.image_paths = [
            os.path.join(root_dir, img) 
            for img in os.listdir(root_dir) 
            if os.path.splitext(img)[1].lower() in self.image_extensions
        ]
        self.image_names = [str(pathlib.Path(os.path.basename(path)).stem) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.image_names[idx]