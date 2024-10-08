from pathlib import Path

from torch.optim import SGD

from dataloaders.face_recognition import create_face_recognition_dataloader
from nets.arcface import create_face_recognition_model
from trainers.arcface import ArcFaceTrainer
from utils.utils import init_logging_configs

TITLE: str = 'face_recog'
DATA_ROOT_DIR: str = 'datasets/celeba-recog-3'
DEVICE: str = 'cpu'
BATCH_SIZE: int = 64
EPOCHS: int = 50
LEARNING_RATE: float = 0.01
NUM_WORKERS: int = 0
IMAGE_SIZE: int = 256
EMBEDDING_DIM: int = 3
MARGIN: float = 0.05
SCALE: float = 30.0
DROPOUT: float = 0.0
DEBUGGING = False

init_logging_configs(DEBUGGING)
data_dir = Path(DATA_ROOT_DIR)
train_loader = create_face_recognition_dataloader(
    root_dir=data_dir/'train', 
    image_size=IMAGE_SIZE,
    augmentation=False,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)
val_loader = create_face_recognition_dataloader(
    root_dir=data_dir/'val', 
    image_size=IMAGE_SIZE,
    augmentation=False,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)
model = create_face_recognition_model(train_loader.dataset.num_classes, EMBEDDING_DIM, dropout=DROPOUT)
optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

trainer = ArcFaceTrainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_classes=train_loader.dataset.num_classes,
    device=DEVICE,
    mission_name=TITLE,
    debugging=DEBUGGING
)

trainer.fit(EPOCHS)