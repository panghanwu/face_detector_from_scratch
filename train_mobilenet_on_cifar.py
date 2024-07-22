from torch import nn
from torch.optim import SGD

from dataloaders.cifar100 import get_cifar100_datasets
from nets.mobilenet import create_mobilenet_for_cifar
from trainers.classification import ClassificationTrainer
from utils import init_logging_configs

CIFAR100_DIR: str = 'datasets'
DEVICE: str = 'cpu'
BATCH_SIZE: int = 64
EPOCHS: int = 200
DROPOUT: float = 0.1
DEBUGGING = False

init_logging_configs(DEBUGGING)
model = create_mobilenet_for_cifar(num_classes=100)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
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