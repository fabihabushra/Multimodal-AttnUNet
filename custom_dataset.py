# custom_dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DualModalityDataset(Dataset):
    def __init__(self, root_dir, images_t1c_dir, images_t2f_dir, masks_dir, img_names, h, w,
                 mean_t1c, std_t1c, mean_t2f, std_t2f, return_path=False):
        self.root_dir = root_dir
        self.images_t1c_dir = images_t1c_dir
        self.images_t2f_dir = images_t2f_dir
        self.masks_dir = masks_dir
        self.img_names = img_names
        self.h = h
        self.w = w
        self.return_path = return_path

        # Transform for t1c modality
        self.transform_t1c = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_t1c, std=std_t1c)
        ])

        # Transform for t2f modality
        self.transform_t2f = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_t2f, std=std_t2f)
        ])

        # Transform for masks (assuming masks are single-channel)
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((self.h, self.w), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_t1c_path = os.path.join(self.root_dir, self.images_t1c_dir, img_name)
        img_t2f_path = os.path.join(self.root_dir, self.images_t2f_dir, img_name)
        mask_path = os.path.join(self.root_dir, self.masks_dir, img_name)

        # Load images
        image_t1c = Image.open(img_t1c_path)#.convert('L')  # Convert to grayscale
        image_t1c = np.array(image_t1c)
        image_t2f = Image.open(img_t2f_path)#.convert('L')
        image_t2f = np.array(image_t2f)
        mask = Image.open(mask_path)#.convert('L')  # Assuming masks are grayscale
        mask = np.array(mask)
        

        # Apply transformations
        image_t1c = self.transform_t1c(image_t1c)
        image_t2f = self.transform_t2f(image_t2f)
        mask = self.mask_transform(mask)
        mask = mask.long() 

        if self.return_path:
            return (image_t1c, image_t2f), mask, img_name
        else:
            return (image_t1c, image_t2f), mask
