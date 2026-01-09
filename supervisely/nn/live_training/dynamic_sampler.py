import random
import time
from collections.abc import Sized


class DynamicSampler:
    """
    A sampler that dynamically adjusts to the size of a dataset that grows over time.
    Implements torch.utils.data.Sampler interface.
    """
    
    def __init__(self, dataset: Sized, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        remaining_indices = []
        last_known_len = 0
        
        # Wait for first samples
        while len(self.dataset) == 0:
            time.sleep(0.1)
        
        while True:
            current_len = len(self.dataset)
            
            if current_len > last_known_len:
                new_indices = list(range(last_known_len, current_len))
                if self.shuffle:
                    random.shuffle(new_indices)
                remaining_indices.extend(new_indices)
                last_known_len = current_len
            
            if not remaining_indices:
                # Reshuffle existing data
                remaining_indices = list(range(current_len))
                if self.shuffle:
                    random.shuffle(remaining_indices)
            
            yield remaining_indices.pop()
    
    def __len__(self):
        return len(self.dataset)