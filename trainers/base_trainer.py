import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from torch import Tensor
from torch.nn import Conv2d, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class CheckpointHandler:
    def __init__(
        self, 
        patience: int, 
        save_dir: str, 
    ) -> None:
        self.patience = patience
        self.counter = 0
        self.best_metric = None
        self.best_epoch = None
        self.root = Path(save_dir)
        self.last_ckp = None
        self.best_ckp = None
        self.stopping = False

        if patience != 0:
            logging.info(f'Set early stopping with patience of {patience} epochs.')

    def __call__(
        self, 
        metric: float, 
        epoch_i: int, 
        checkpoint: dict,
        prefer_lower: bool,
    ) -> bool:
        self.remove_checkpoint(self.last_ckp)
        self.last_ckp = self.save_checkpoint(epoch_i, 'last', checkpoint)

        if self.is_best(metric, prefer_lower):
            self.remove_checkpoint(self.best_ckp)
            self.best_ckp = self.save_checkpoint(epoch_i, 'best', checkpoint)
            logging.info(f'Best checkpoint {self.best_ckp} saved.')
            self.best_epoch = epoch_i
            self.best_metric = metric
            self.counter = 0
        elif self.patience != 0:
            self.counter += 1
            self.stopping = self.counter == self.patience
        return self.stopping

    @staticmethod
    def remove_checkpoint(path: Optional[Path]) -> None:
        try:
            path.unlink()
        except AttributeError:
            if path is not None:
                path.unlink()  # reproduce the error
        except FileNotFoundError:
            logging.warning(f'The previous checkpoint is not found: {path}')

    def save_checkpoint(self, epoch_i: int, prefix: str, checkpoint: dict) -> Path:
        path = self.root / f'{prefix}-epoch{epoch_i}.pth'
        torch.save(checkpoint, path)
        return path
    
    def is_best(self, metric: float, prefer_lower: bool) -> bool:
        try:
            if prefer_lower and metric < self.best_metric:
                return True
            elif not prefer_lower and metric > self.best_metric:
                return True
            else:
                return False
        except TypeError:
            if self.best_metric is None:
                self.best_metric = metric
            else:
                self.best_metric == metric  # reproduce the error


class ConvWeightDeltaCalculator:
    def __init__(self, module: Module) -> None:
        self.origin = self.collect_layers_weights(module)

    def __call__(self, module: Module) -> dict[str, tuple[float, float]]:
        measurement = {}
        self.this = self.collect_layers_weights(module)
        for name in self.origin.keys():
            delta = torch.abs(self.this[name] - self.origin[name])
            measurement[name] = (delta.mean().item(), delta.std().item())
        return measurement
    
    def collect_convs_weights(self, module: Module) -> list[Tensor]:
        weights = []
        for child in module.children():
            if isinstance(child, Conv2d):
                weights.append(child.weight.data)
            else:
                weights.extend(self.collect_convs_weights(child))
        return weights

    def collect_layers_weights(self, module: Module) -> dict[str, Tensor]:
        layers = {}
        for name, child in module.named_children():
            weights = self.collect_convs_weights(child)
            flatten_weights = [w.flatten() for w in weights]
            layers[name] = torch.cat(flatten_weights)
        return layers
    
    @property
    def num_params(self) -> dict[str, int]:
        counter = {}
        for name, weights in self.origin.items():
            counter[name] = weights.numel()
        return counter


class BaseTrainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Callable,
        device: str = 'cpu',
        configs: Optional[dict] = None,
        tensor_dtype: torch.dtype = torch.float32,
        mission_name: str = 'train',
        stopping_patience: int = 0,
        debugging: bool = False
    ) -> None:
        self.debugging = debugging
        self.tensor_cfgs = {'device': torch.device(device), 'dtype': tensor_dtype}
        self.model = model.to(self.tensor_cfgs['device'])
        self.criterion = criterion
        self.optimizer = optimizer
        self.configs = configs
        self.dataloaders = {'train': train_loader, 'val': val_loader}

        self.root = self._create_log_dir(mission_name, 'train')
        self.tensorboard = self.init_tensorboard(self.root)
        self.ckpt_handler = CheckpointHandler(stopping_patience, self.root / 'checkpoints')
        self._reset_epoch_log_keeper()

    @staticmethod
    def init_tensorboard(logdir: Path):
        tb = SummaryWriter(logdir / 'tensorboard')
        logging.info(f'📊 Tensorboard command line: tensorboard --logdir {logdir.absolute()}')
        return tb

    @staticmethod 
    def _create_log_dir(name: str, parent: str = '') -> Path:
        parent = Path(parent)
        log_dir = parent / name
        index = 0
        while log_dir.exists():
            index += 1
            log_dir = parent / f'{name}-{index}'
        (log_dir / 'checkpoints').mkdir(parents=True)
        logging.info(f'📂 Training log directory: {log_dir}')
        return log_dir
    
    @staticmethod
    def _add_data_to_csv(path: str, data: dict) -> None:
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

    def _update_csv_log(self):
        csv_data = {'epoch': self.epoch_i}
        for name in self.epoch_logs.keys():
            for phase in self.epoch_logs[name].keys():
                csv_data[f'{name}/{phase}'] = self.epoch_logs[name][phase]
        self._add_data_to_csv(self.root / 'log.csv', csv_data)

    def _update_scalars_to_tensorboard(self):
        for k, v in self.epoch_logs.items():
            self.tensorboard.add_scalars(k, v, self.epoch_i)

    def _reset_epoch_log_keeper(self) -> None:
        self.epoch_logs = defaultdict(dict[Literal['train', 'val'], float])

    def finish_phase(
        self, 
        phase: Literal['train', 'val'], 
        accumulation: dict[str, float], 
        total_num_data: Optional[int] = None
    ) -> None:
        divider = total_num_data if total_num_data is not None else 1.
        for k, v in accumulation.items():
            self.epoch_logs[k][phase] = v / divider
    
    def cook_epoch_info(self) -> str:
        # customize the info
        info = f'| train_loss {self.epoch_logs["loss"]["train"]:.2e} | val_loss {self.epoch_logs["loss"]["val"]:.2e}'
        return info

    def finish_epoch(self) -> None:
        self._update_csv_log()
        self._update_scalars_to_tensorboard()
        
        epoch_info = self.cook_epoch_info()
        logging.info(f'Epoch {self.epoch_i} {epoch_info}')
        self._reset_epoch_log_keeper()
        self.epoch_i += 1
        
    def load_batch(self, batch):
        """Example
        inp = batch['input'].to(**self.tensor_cfgs)
        tar = batch['target'].to(**self.tensor_cfgs)
        msk = batch['mask'].to(self.tensor_cfgs['device'])
        return inp, tar, msk
        """
        raise NotImplementedError

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
                        _ = self.load_batch(batch)  # TODO

                        if phase == 'train':
                            self.optimizer.zero_grad()

                        out = self.model(_)  # TODO
                        loss: Tensor = self.criterion(_)  # TODO

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                                                
                        accumulator['loss'] += loss.item()

                        batch_loss = loss.item() / self.dataloaders[phase].batch_size
                        data_for_tqdm = {'loss': f'{batch_loss:.2e}'}
                        dataloader.set_postfix(data_for_tqdm)

                    total_num_data = len(self.dataloaders[phase].dataset)
                    self.finish_phase(phase, accumulator, total_num_data)
                    dataloader.close()

            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'configs': self.configs
            }
            early_stopping = self.ckpt_handler(self.epoch_logs['loss']['val'], 
                                               self.epoch_i, checkpoint, prefer_lower=True)
            self.finish_epoch()
            if early_stopping:
                logging.info(f'Early stopping at epoch {self.epoch_i}.')
                break

        logging.info(f'=== Mission completed. 🦾 ===')
        raise NotImplementedError