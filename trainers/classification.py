import logging
from collections import defaultdict
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(
        self, 
        model: Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: Optimizer, 
        criterion: Callable, 
        device: str = 'cpu', 
        configs: dict | None = None, 
        tensor_dtype: torch.dtype = torch.float32, 
        mission_name: str = 'train',
        debugging: bool = False
    ) -> None:
        super().__init__(model, train_loader, val_loader, optimizer, 
                         criterion, device, configs, 
                         tensor_dtype, mission_name, debugging)

    @torch.no_grad    
    def count_correct(self, output: Tensor, target: Tensor) -> int:
        _, predictions = torch.max(output, dim=1)
        return torch.sum(predictions == target).item()
    
    def cook_epoch_info(self) -> str:
        # customize the info 
        info = f'| val_loss {self.epoch_logs["loss"]["val"]:.2e} '
        info += f'| val_acc {self.epoch_logs["accuracy"]["val"]:.0%} '
        return info
    
    def load_batch(self, batch) -> tuple[Tensor, Tensor]:
        inp = batch[0].to(**self.tensor_cfgs)
        tar = batch[1].to(self.tensor_cfgs['device'])
        return inp, tar

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
                        inp, tar = self.load_batch(batch)

                        if phase == 'train':
                            self.optimizer.zero_grad()

                        out = self.model(inp)
                        loss: Tensor = self.criterion(out, tar)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        
                        accumulator['loss'] += loss.item()
                        batch_correct = self.count_correct(out, tar)
                        accumulator['accuracy'] += batch_correct

                        batch_loss = loss.item() / len(tar)
                        batch_acc = batch_correct / len(tar)
                        data_for_tqdm = {'loss': f'{batch_loss:.2e}', 'acc': f'{batch_acc:.2f}'}
                        dataloader.set_postfix(data_for_tqdm)
                    
                    total_num_data = len(self.dataloaders[phase].dataset)
                    self.finish_phase(phase, accumulator, total_num_data)
                    dataloader.close()

            self.finish_epoch(self.epoch_logs['loss']['val'])

        logging.info(f'=== Mission completed. 🦾 ===')

        logging.info(f'Best epoch: {self._ckpt_meta["best_epoch"]}')