import numpy as np
from typing import Callable


class LossPlateauDetector:
    """
    Detect plateau in training loss using moving average comparison.
    
    Args:
        window_size: Number of iterations for moving average
        threshold: Relative change threshold (e.g., 0.005 = 0.5%)
        patience: Number of consecutive plateau detections before action
        check_interval: Check frequency (every N iterations)
    """

    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 0.005,
        patience: int = 1,
        check_interval: int = 1,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.check_interval = check_interval
        self.patience = patience
        self._min_iterations = 2 * window_size

        # State
        self.loss_history = []
        self.consecutive_plateau_count = 0
        self._save_checkpoint_fn = None
    
    def register_save_checkpoint_callback(self, fn: Callable[[], None]):
        """Register callback function to save checkpoint when plateau detected"""
        self._save_checkpoint_fn = fn

    def reset(self):
        """Reset detector state"""
        self.loss_history = []
        self.consecutive_plateau_count = 0

    def step(self, loss: float, current_iter: int) -> bool:
        """
        Process one training iteration.
        
        Args:
            loss: Current loss value
            current_iter: Current iteration number
            
        Returns:
            True if plateau confirmed and checkpoint saved, False otherwise
        """
        self.loss_history.append(loss)

        # Check only at specified intervals
        if (current_iter + 1) % self.check_interval != 0:
            return False
        
        # Need enough data
        if len(self.loss_history) < self._min_iterations:
            return False
        
        # Check for plateau
        is_plateau, info = self._check_plateau(current_iter)
        
        if is_plateau:
            self.consecutive_plateau_count += 1
            print(
                f'[Plateau Detection] Iteration {current_iter}: '
                f'Signal {self.consecutive_plateau_count}/{self.patience} '
                f'(change: {info["metric"]:.6f}, threshold: {self.threshold})'
            )
            
            # Trigger action when patience reached
            if self.consecutive_plateau_count >= self.patience:
                print(f'[Plateau Detection] Plateau confirmed, saving checkpoint...')
                
                if self._save_checkpoint_fn is not None:
                    self._save_checkpoint_fn()
                    print(f'[Plateau Detection] Checkpoint saved')
                else:
                    print(f'[Plateau Detection] No callback registered')
                
                self.consecutive_plateau_count = 0
                return True
        
        return False
    
    def _check_plateau(self, current_iter: int) -> tuple:
        """Check if current window shows plateau"""
        # Current window average
        current_window = self.loss_history[-self.window_size:]
        current_avg = np.mean(current_window)
        
        # Previous window average
        previous_window = self.loss_history[-2*self.window_size:-self.window_size]
        previous_avg = np.mean(previous_window)
        
        change = previous_avg - current_avg
        is_plateau = change < self.threshold
        
        info = {
            'iter': current_iter,
            'metric': change,
            'threshold': self.threshold,
            'previous_avg': previous_avg,
            'current_avg': current_avg,
        }
        
        return is_plateau, info