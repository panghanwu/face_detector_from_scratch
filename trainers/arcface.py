import logging
from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
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

    @torch.no_grad
    def count_correct(self, output: Tensor, target: Tensor) -> int:
        _, predictions = torch.max(output, dim=1)
        return torch.sum(predictions == target).item()
    
    def cook_epoch_info(self) -> str:
        # customize the info 
        info = f'| train_loss {self.epoch_logs["loss"]["train"]:.2e} '
        info += f'| val_loss {self.epoch_logs["loss"]["val"]:.2e} '
        info += f'| val_acc {self.epoch_logs["accuracy"]["val"]:.0%} '
        return info
    
    def load_batch(self, batch) -> tuple[Tensor, Tensor]:
        images = batch[0].to(**self.tensor_cfgs)
        labels = batch[1].to(self.tensor_cfgs['device'])
        return images, labels

    def fit(self, epochs: Optional[int] = None):

        if epochs is None:
            logging.info(f'=== Endless mission starts! 🛸 ===')
            epochs = torch.inf
        else:
            logging.info(f'=== Mission starts! 🚀 ===')
        
        self.epoch_i = 0
        while self.epoch_i < epochs:

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

                        output = self.model(images)
                        loss: Tensor = self.criterion(output, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        
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

            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'configs': self.configs
            }
            early_stopping = self.ckpt_handler(self.epoch_logs['accuracy']['val'], 
                                               self.epoch_i, checkpoint, prefer_lower=False)
            self.finish_epoch()
            if early_stopping:
                logging.info(f'Early stopping at epoch {self.epoch_i}.')
                break

        logging.info(f'=== Mission completed. 🦾 ===')

        logging.info(f'Best epoch: {self.ckpt_handler.best_epoch}')