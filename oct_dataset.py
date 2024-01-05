import os
import re
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

import torch
import torchvision
import matplotlib.pyplot as plt


class OCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.scans = [os.path.join(root_dir, scan)
                      for scan in os.listdir(root_dir)]
        self.seg_paths = self.get_seg_paths()

    def get_seg_paths(self):
        seg_paths = []
        for scan in self.scans:
            segs_path = os.path.join(scan, "segmentation")
            seg_paths += [os.path.join(segs_path, seg)
                          for seg in os.listdir(segs_path)]
        return seg_paths

    def __len__(self):
        return len(self.seg_paths)

    def __getitem__(self, idx):
        seg_path = self.seg_paths[idx]
        scan_path, file_name = seg_path.split('\\segmentation\\')
        image_path = os.path.join(scan_path, file_name)

        with Image.open(image_path) as img:
            if self.transform:
                img = self.transform(img)

        with Image.open(seg_path) as segmentation:
            if self.transform:
                segmentation = self.transform(segmentation)

        return img, segmentation
