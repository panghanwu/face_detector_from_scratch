from torch import nn
from torch.optim import Adam

from dataloaders.cifar100 import get_cifar100_datasets
from nets.mobilenet import create_mobilenet_large_for_classification
from trainers.classification import ClassificationTrainer
from utils import init_logging_configs

CIFAR100_DIR: str = 'datasets'
DEVICE: str = 'cpu'
BATCH_SIZE: int = 64
EPOCHS: int = 100
LEARNING_RATE: float = 0.0001
DEBUGGING = False

init_logging_configs(DEBUGGING)
model = create_mobilenet_large_for_classification(num_classes=100)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
train_loader, test_loader = get_cifar100_datasets(BATCH_SIZE, CIFAR100_DIR)

trainer = ClassificationTrainer(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion=nn.CrossEntropyLoss(),
    device=DEVICE,
    mission_name='cifar_mobilenet',
    debugging=DEBUGGING
)

trainer.fit(EPOCHS)