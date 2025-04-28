# python/alphazero/training/scheduler.py
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    """Learning rate scheduler with warmup and cosine annealing.
    
    This scheduler combines linear warmup with cosine annealing for
    a smooth learning rate schedule throughout training.
    
    Args:
        optimizer (Optimizer): Optimzer to use
        warmup_epochs (int): Number of epochs for warmup phase
        max_epochs (int): Total number of epochs for training
        min_lr (float, optional): Minimum learning rate. Default: 0
        last_epoch (int, optional): The index of last epoch. Default: -1
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(1.0, progress)  # Ensure progress <= 1
            
            return [self.min_lr + 0.5 * (base_lr - self.min_lr) * 
                    (1 + math.cos(math.pi * progress)) 
                    for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """Learning rate scheduler with linear warmup phase.
    
    Args:
        optimizer (Optimizer): Optimzer to use
        warmup_epochs (int): Number of epochs for warmup phase
        after_scheduler (LRScheduler): Scheduler to use after warmup
        last_epoch (int, optional): The index of last epoch. Default: -1
    """
    
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished_warmup = True
            
            return self.after_scheduler.get_lr()
            
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        if epoch < self.warmup_epochs:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished_warmup = True
            
            self.after_scheduler.step()


class CyclicCosineAnnealingLR(_LRScheduler):
    """Cyclic cosine annealing learning rate scheduler.
    
    This scheduler uses cosine annealing with restarts, which helps
    the model escape local minima during training.
    
    Args:
        optimizer (Optimizer): Optimzer to use
        cycle_epochs (int): Number of epochs in each cycle
        min_lr (float, optional): Minimum learning rate. Default: 0
        cycle_mult (float, optional): Multiplier for cycle length after each cycle. Default: 1.0
        last_epoch (int, optional): The index of last epoch. Default: -1
    """
    
    def __init__(self, optimizer, cycle_epochs, min_lr=0, cycle_mult=1.0, last_epoch=-1):
        self.cycle_epochs = cycle_epochs
        self.min_lr = min_lr
        self.cycle_mult = cycle_mult
        self.cycle = 0
        self.cycle_epoch = 0
        super(CyclicCosineAnnealingLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        # Calculate where we are in the current cycle
        current_cycle_len = self.cycle_epochs * (self.cycle_mult ** self.cycle)
        progress = self.cycle_epoch / current_cycle_len
        
        # Check if we need to start a new cycle
        if progress >= 1.0:
            self.cycle += 1
            self.cycle_epoch = 0
            current_cycle_len = self.cycle_epochs * (self.cycle_mult ** self.cycle)
            progress = 0
        
        # Apply cosine annealing within the cycle
        return [self.min_lr + 0.5 * (base_lr - self.min_lr) * 
                (1 + math.cos(math.pi * progress)) 
                for base_lr in self.base_lrs]
            
    def step(self, epoch=None):
        if epoch is None:
            self.cycle_epoch += 1
        else:
            # Calculate cycle and cycle_epoch from absolute epoch
            total_epochs = 0
            cycle = 0
            while True:
                cycle_len = self.cycle_epochs * (self.cycle_mult ** cycle)
                if total_epochs + cycle_len > epoch:
                    self.cycle = cycle
                    self.cycle_epoch = epoch - total_epochs
                    break
                total_epochs += cycle_len
                cycle += 1
        
        super(CyclicCosineAnnealingLR, self).step()