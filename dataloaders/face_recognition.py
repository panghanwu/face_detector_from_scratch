from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import Tensor
import torch
import cv2
import pandas as pd

from utils.tensor_utils import pad_to_square


class FaceRecognitionData(Dataset):
    """
    Directory structure:
        dataset/
        ├── images/
        │   ├── 001.jpg
        │   ├── 002.jpg
        │   └── ...
        └── mapping.csv

    Structure of mapping.txt
        image_file_name, id
    """
    def __init__(
        self, 
        dataset_dir: str, 
        image_size: int = 256, 
        augmentation: bool = False
    ) -> None:
        self.root = Path(dataset_dir)
        self.mapping = pd.read_csv(self.root / 'mapping.csv', index_col=0, header=None)
        self.num_classes = self.mapping[1].max() + 1
        self.mapping = self.mapping[1].to_dict()
        self.image_list = list(self.mapping.keys())
        self.img_sz = (image_size, image_size)
        self.aug = augmentation
        self.to_tensor = transforms.ToTensor()

        if augmentation:
            crop_padding = int(0.3*image_size)
            self.transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                transforms.RandomRotation(degrees=20),
                transforms.RandomCrop(size=image_size, padding=crop_padding)
            ])

    def __len__(self):
        return len(self.mapping)
    
    def __getitem__(self, index) -> tuple[Tensor, int]:
        fn = self.image_list[index]
        label = self.mapping[fn]
        image_path = (self.root / 'images') / fn
        img = cv2.imread(str(image_path))[..., ::-1]
        img = pad_to_square(img, fill=0)
        img = cv2.resize(img, self.img_sz)
        img = self.to_tensor(img)
        if self.aug:
            img = self.transforms(img)

        return img, label


def create_face_recognition_dataloader(
    root_dir: str, 
    image_size: int = 256,
    augmentation: bool = False,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    dataset = FaceRecognitionData(root_dir, image_size, augmentation)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )