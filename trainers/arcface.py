import logging
from collections import defaultdict
from copy import copy
from typing import Optional
import csv

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.arcface import ArcFaceLoss
from .base_trainer import BaseTrainer


class ArcFaceTrainer(BaseTrainer):
    def __init__(
        self, 
        model: Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: Optimizer, 
        num_classes: int,
        device: str = 'cpu', 
        margin: float = 0.1,
        scale: float = 1.0,
        configs: dict | None = None, 
        tensor_dtype: torch.dtype = torch.float32, 
        mission_name: str = 'train',
        stopping_patience: int = 0,
        debugging: bool = False
    ) -> None:
        super().__init__(model, train_loader, val_loader, optimizer, 
                         None, device, configs, 
                         tensor_dtype, mission_name, stopping_patience, 
                         debugging)
        self.criterion = ArcFaceLoss(num_classes, margin, scale)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=10)

    @torch.no_grad
    def count_correct(self, output: Tensor, target: Tensor) -> int:
        _, predictions = torch.max(output, dim=1)
        return torch.sum(predictions == target).item()
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
    
    def cook_epoch_info(self) -> str:
        # customize the info 
        info = f'| get_lr {self.get_lr():.1e} '
        info += f'| train_loss {self.epoch_logs["loss"]["train"]:.3e} '
        info += f'| val_loss {self.epoch_logs["loss"]["val"]:.3e} '
        info += f'| val_acc {self.epoch_logs["accuracy"]["val"]:.0%} '
        return info
    
    def load_batch(self, batch) -> tuple[Tensor, Tensor]:
        images = batch[0].to(**self.tensor_cfgs)
        labels = batch[1].to(self.tensor_cfgs['device'])
        return images, labels
    
    def add_embeddings_to_csv(self, path: str, embeddings: list[list[float]], groundtruth: list[int]) -> None:
        data = [[self.epoch_i, gt] + em for em, gt in zip(embeddings, groundtruth)]
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def fit(self, epochs: Optional[int] = None):

        if epochs is None:
            logging.info(f'=== Endless mission starts! 🛸 ===')
            epochs = torch.inf
        else:
            logging.info(f'=== Mission starts! 🚀 ===')
        
        self.epoch_i = 0
        while self.epoch_i < epochs:
            embeddings = []
            groundtruth = []

            for phase in ['train', 'val']:
                self.model.train(phase=='train')
                dataloader = tqdm(
                    self.dataloaders[phase], 
                    desc=f'Epoch {self.epoch_i} - {phase}', 
                    leave=False
                )
                accumulator = defaultdict(float)
                
                with torch.set_grad_enabled(phase=='train'):

                    for batch_i, batch in enumerate(dataloader):

                        if self.debugging and batch_i > 10: break

                        # load batch data
                        images, labels = self.load_batch(batch)

                        if phase == 'train':
                            self.optimizer.zero_grad()

                        embedding = self.model[0](images)
                        output = self.model[1](embedding)
                        loss: Tensor = self.criterion(output, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                        embeddings += embedding.tolist()
                        groundtruth += labels.tolist()
                        
                        accumulator['loss'] += loss.item()
                        batch_correct = self.count_correct(output, labels)
                        accumulator['accuracy'] += batch_correct

                        batch_loss = loss.item() / len(labels)
                        batch_acc = batch_correct / len(labels)
                        data_for_tqdm = {'loss': f'{batch_loss:.2e}', 'acc': f'{batch_acc:.2f}'}
                        dataloader.set_postfix(data_for_tqdm)
                    
                    total_num_data = len(self.dataloaders[phase].dataset)
                    self.finish_phase(phase, accumulator, total_num_data)
                    dataloader.close()

            self.add_embeddings_to_csv(self.root / 'embeddings.csv', embeddings, groundtruth)
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'configs': self.configs
            }
            train_loss = copy(self.epoch_logs['loss']['train'])
            early_stopping = self.ckpt_handler(self.epoch_logs['loss']['val'], self.epoch_i, checkpoint, prefer_lower=True)
            self.finish_epoch()
            self.scheduler.step(train_loss)
            if early_stopping:
                logging.info(f'Early stopping at epoch {self.epoch_i}.')
                break

        logging.info(f'=== Mission completed. 🦾 ===')

        logging.info(f'Best epoch: {self.ckpt_handler.best_epoch}')