import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
        debugging: bool = False
    ) -> None:
        self.debugging = debugging
        self.init_logging_configs(self.debugging)
        self.tensor_cfgs = {'device': torch.device(device), 'dtype': tensor_dtype}
        self.model = model.to(self.tensor_cfgs['device'])
        self.criterion = criterion
        self.optimizer = optimizer
        self.configs = configs
        self.dataloaders = {'train': train_loader, 'val': val_loader}

        self.root = self._create_log_dir(mission_name, 'train')
        self.tensorboard = self.init_tensorboard(self.root)
        self._ckpt_meta = {
            'dir': self.root / 'checkpoints', 'best_metric': torch.inf, 
            'best_epoch': None, 'best_fn': None, 'last_fn': None}
        self._reset_epoch_log_keeper()
    
    @staticmethod
    def init_logging_configs(debugging: bool = False) -> None:
        """Add preinformation here"""
        if debugging:
            logging.basicConfig(
                format='%(asctime)s | %(levelname)s | %(filename)s | %(message)s',
                level=logging.DEBUG
            )
            logging.info('☢️ Debugging mode!!!')
        else:
            logging.basicConfig(
                format='%(asctime)s | %(levelname)s | %(message)s',
                level=logging.INFO
            )

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
        with open(path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

    @staticmethod
    def _remove_checkpoint(path: Optional[Path]) -> None:
        try:
            path.unlink()
        except AttributeError:
            if path is not None:
                path.unlink()
        except FileNotFoundError:
            logging.warning(f'The previous checkpoint is not found: {path}')

    def save_checkpoint(self, metric: float) -> None:
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'configs': self.configs
        }
        
        self._remove_checkpoint(self._ckpt_meta['last_fn'])
        self._ckpt_meta['last_fn'] = self._ckpt_meta['dir'] / f'last-epoch{self.epoch_i}.pth'
        torch.save(checkpoint, self._ckpt_meta['last_fn'])
        
        if self._ckpt_meta['best_metric'] > metric:
            self._ckpt_meta['best_metric'] = metric
            self._remove_checkpoint(self._ckpt_meta['best_fn'])
            self._ckpt_meta['best_fn'] = self._ckpt_meta['dir'] / f'best-epoch{self.epoch_i}.pth'
            self._ckpt_meta['best_epoch'] = self.epoch_i
            torch.save(checkpoint, self._ckpt_meta['best_fn'])
            logging.info(f'Best checkpoint {self._ckpt_meta["best_fn"]} saved.')

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

    def finish_epoch(self, val_metric: float) -> None:
        self._update_csv_log()
        self._update_scalars_to_tensorboard()
        
        epoch_info = self.cook_epoch_info()
        logging.info(f'Epoch {self.epoch_i} {epoch_info}')
        self.save_checkpoint(val_metric)
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

            self.finish_epoch(self.epoch_logs['loss']['val'])

        logging.info(f'=== Mission completed. 🦾 ===')
        raise NotImplementedError