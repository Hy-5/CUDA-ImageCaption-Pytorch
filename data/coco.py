from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import json
import os
from PIL import Image


class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, json_path, img_folder, transform=None):
        """
        Args:
            json_path (string): path to COCO annotation file
            img_folder (string): folder where COCO images are stored
            transform (callable, optional): optional transform to be applied
                on a sample.
        """
        with open(json_path, 'r') as j:
            self.annotations = json.load(j)
        self.img_folder = img_folder
        self.transform = transform

    def __getitem__(self, index):
        annotation = self.annotations['annotations'][index]
        img_id = annotation['image_id']
        img_name = os.path.join(self.img_folder, '%012d.jpg' % img_id)
        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = torch.Tensor(annotation['caption'])
        return image, target

    def __len__(self):
        return len(self.annotations['annotations'])
