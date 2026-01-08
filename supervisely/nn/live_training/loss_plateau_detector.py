import numpy as np
from typing import Callable


class LossPlateauDetector():
    """Detect plateau in training loss using moving average comparison.
    
    This hook monitors the training loss and detects when it has plateaued
    by comparing the moving average of recent losses with the previous window.
    
    Args:
        window_size (int): Number of iterations to use for computing the 
            moving average. Default: 50.
        threshold (float): Relative change threshold below which plateau is 
            detected (e.g., 0.005 means 0.5% change). Default: 0.005.
        patience (int): Number of consecutive plateau detections required before
            taking action. This prevents premature stopping due to temporary
            fluctuations. Default: 1.
        check_interval (int): How often to check for plateau (every N iterations).
            If None, defaults to window_size.
            Default: None.
    
    """

    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 0.005,
        patience: int = 1,
        check_interval: int = 1,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.check_interval = check_interval or window_size
        self.patience = patience
        self._min_iterations = 2 * window_size

        # State variables
        self.loss_history = []
        self.consecutive_plateau_count = 0
    
    def register_save_checkpoint_callback(self, fn: Callable[[], None]) -> None:
        """Register a callback function to save checkpoints."""
        self._save_checkpoint_fn = fn

    def reset(self) -> None:
        """Reset the detector state."""
        self.loss_history = []
        self.consecutive_plateau_count = 0
        self._min_iterations = 2 * self.window_size

    def step(
        self,
        loss: float,
        current_iter: int,
        runner=None,
    ) -> bool:        
        self.loss_history.append(loss)

        # Only check at specified intervals
        if (current_iter + 1) % self.check_interval != 0:
            return
        
        # Check if we have enough data
        if len(self.loss_history) < self._min_iterations:
            return
        
        # Perform plateau detection
        is_plateau, info = self._check_plateau(current_iter)

        # Handle plateau detection
        if is_plateau:
            self.consecutive_plateau_count += 1
            print(
                f'[Plateau Detection] Iteration {current_iter}: '
                f'Plateau signal {self.consecutive_plateau_count}/{self.patience}\n'
                f'    Change: {info["metric"]:.6f} '
                f'(Previous Avg: {info["previous_avg"]:.6f}, Current Avg: {info["current_avg"]:.6f}) '
                f'(threshold: {self.threshold})'
            )
            
            # Check if we've reached patience threshold
            if self.consecutive_plateau_count >= self.patience:
                print(f'[Plateau Detection] Plateau confirmed! Triggering checkpoint save...')
                
                if runner is not None:
                    # Find CheckpointHook in runner's hooks
                    checkpoint_hook = None
                    for hook in runner.hooks:
                        if hook.__class__.__name__ == 'CheckpointHook':
                            checkpoint_hook = hook
                            break
                    
                    if checkpoint_hook is not None:
                        # Temporarily override filename
                        original_filename = checkpoint_hook.filename_tmpl
                        checkpoint_hook.filename_tmpl = f'plateau_iter_{current_iter}.pth'
                        
                        # Trigger checkpoint save through the hook
                        self._save_checkpoint_fn()
                        
                        # Restore original filename
                        checkpoint_hook.filename_tmpl = original_filename
                        
                        print(f'[Plateau Detection] Checkpoint saved via CheckpointHook')
                    else:
                        print(f'[Plateau Detection] CheckpointHook not found, cannot save checkpoint')
                else:
                    print(f'[Plateau Detection] Runner not available, checkpoint not saved')
                
                # Reset counter after action
                self.consecutive_plateau_count = 0
                return True
    
    def _check_plateau(self, current_iter: int) -> tuple:
        # Calculate current window average (last N steps)
        current_window = self.loss_history[-self.window_size:]
        current_avg = np.mean(current_window)
        
        # Calculate previous window average (N steps before current window)
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