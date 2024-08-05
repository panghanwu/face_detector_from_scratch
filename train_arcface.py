from pathlib import Path

from torch.optim import Adam

from dataloaders.face_recognition import create_face_recognition_dataloader
from nets.arcface import create_face_recognition_model
from trainers.arcface import ArcFaceTrainer
from utils.utils import init_logging_configs


DATA_ROOT_DIR: str = 'datasets/celeba-recog-50'
DEVICE: str = 'cpu'
BATCH_SIZE: int = 16
EPOCHS: int = 200
NUM_WORKERS: int = 0
IMAGE_SIZE: int = 256
EMBEDDING_DIM: int = 3
DEBUGGING = False

init_logging_configs(DEBUGGING)
data_dir = Path(DATA_ROOT_DIR)
train_loader = create_face_recognition_dataloader(
    root_dir=data_dir/'train', 
    image_size=IMAGE_SIZE,
    augmentation=True,
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
model = create_face_recognition_model(train_loader.dataset.num_classes, EMBEDDING_DIM)
optimizer = Adam(model.parameters(), lr=0.001)

trainer = ArcFaceTrainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    device=DEVICE,
    mission_name='arcface',
    stopping_patience=0,
    debugging=DEBUGGING
)

trainer.fit(EPOCHS)