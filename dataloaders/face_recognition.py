from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import Tensor
from numpy import ndarray
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
        self.img_sz = (image_size, image_size)
        self.aug = augmentation

        df = pd.read_csv(self.root / 'mapping.csv', header=None)
        self.num_classes = df[1].max() + 1
        self.image_list = df[0].tolist()
        self.labels = df[1].tolist()
        
        if augmentation:
            crop_padding = int(0.3*image_size)
            self.transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                transforms.RandomRotation(degrees=20),
                transforms.RandomCrop(size=image_size, padding=crop_padding)
            ])

    def __len__(self):
        return len(self.labels)
    
    @staticmethod
    def preprocessing(bgr_image: ndarray, size: tuple[int, int]) -> Tensor:
        image = pad_to_square(bgr_image[..., ::-1], fill=0)
        image = cv2.resize(image, size)
        tensor = torch.tensor(image).float().permute(2, 0, 1)
        tensor /= 255.
        return tensor
    
    def __getitem__(self, index) -> tuple[Tensor, int]:
        fn = self.image_list[index]
        label = self.labels[index]
        image_path = (self.root / 'images') / fn
        img = cv2.imread(str(image_path))
        img = self.preprocessing(img, self.img_sz)
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