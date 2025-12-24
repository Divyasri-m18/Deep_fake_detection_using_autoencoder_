import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob

class FaceDataset(Dataset):
    """
    Dataset for loading processed face images.
    Supports recursive search for nested directories (Kaggle dataset).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # Recursive glob for jpg and png
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.image_paths.extend(glob(os.path.join(root_dir, "**", ext), recursive=True))
            
        self.transform = transform
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            # Handle bad images gracefully
            return torch.zeros((3, 128, 128))

def get_transforms(target_size=128, is_train=False):
    """
    Standard transforms: Resize -> Tensor -> Normalize
    """
    if is_train:
        # Stronger augmentations for training (inspired by selimsef)
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(), # [0-1]
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)), # Cutout-like
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])

def get_dataloader(data_dir, batch_size, shuffle=True, num_workers=0, is_train=False):
    dataset = FaceDataset(data_dir, transform=get_transforms(is_train=is_train))
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    # Test
    print("Normalization utils ready.")
