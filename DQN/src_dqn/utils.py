import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class Logger(object):
    
    def __init__(self, log_file = None):
        self.log_file = log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file = self.log_file)
            self.log_file.flush()

class CudaTensorManager(object):
    def __init__(self,
                 state_shape: Tuple,
                 batch_size: int,
                 use_cuda: bool=True,
                 dtype=np.uint8):
        
        # Allocate pinned memory at once
        # states and pinned states are allocated as uint8 to save transfer time
        self.dtype = dtype
        if dtype == np.uint8:
            self.pinned_next_state = torch.empty(batch_size,
                                                 *state_shape,
                                                 dtype=torch.uint8,
                                                 pin_memory=True)
            self.pinned_state = torch.empty(batch_size,
                                            *state_shape,
                                            dtype=torch.uint8,
                                            pin_memory=True)
        else:
            self.pinned_next_state = torch.empty(batch_size,
                                                 *state_shape,
                                                 dtype=torch.float32,
                                                 pin_memory=True)
            self.pinned_state = torch.empty(batch_size,
                                            *state_shape,
                                            dtype=torch.float32,
                                            pin_memory=True)

        self.pinned_reward = torch.empty(batch_size,
                                         dtype=torch.float32,
                                         pin_memory=True)
        self.pinned_done = torch.empty(batch_size,
                                       dtype=torch.float32,
                                       pin_memory=True)
        self.pinned_action = torch.empty(batch_size,
                                         dtype=torch.int64,
                                         pin_memory=True)
        self.use_cuda = use_cuda
        self.ncall = 0

    def get_cuda_tensors(self, state, next_state, action, reward, done):
        """
        Transfer to gpu
        """
        # Copy numpy array to pinned memory
        if self.dtype == np.uint8:
            self.pinned_next_state.copy_(torch.from_numpy(next_state.astype(np.uint8)))
            self.pinned_state.copy_(torch.from_numpy(state.astype(np.uint8)))
        else:
            self.pinned_next_state.copy_(torch.from_numpy(next_state.astype(self.dtype)))
            self.pinned_state.copy_(torch.from_numpy(state.astype(self.dtype)))

        self.pinned_reward.copy_(torch.from_numpy(reward))
        self.pinned_done.copy_(torch.from_numpy(done))
        self.pinned_action.copy_(torch.from_numpy(action))

        # Cuda
        if self.use_cuda:
            # Use asychronous transfer. Order matters, 
            # first is the first tensor we will need to use
            cuda_next_state = self.pinned_next_state.cuda(non_blocking=True)
            cuda_state      = self.pinned_state.cuda(non_blocking=True)
            cuda_reward     = self.pinned_reward.cuda(non_blocking=True)
            cuda_done       = self.pinned_done.cuda(non_blocking=True)
            cuda_action     = self.pinned_action.cuda(non_blocking=True)
        # Not cuda
        else:
            cuda_next_state = self.pinned_next_state
            cuda_state      = self.pinned_state
            cuda_reward     = self.pinned_reward
            cuda_done       = self.pinned_done
            cuda_action     = self.pinned_action

        return cuda_state, cuda_next_state, cuda_action, cuda_reward, cuda_done

def update_target(current_model, target_model, tau: float=None):

    """Soft target network update based on tau."""

    if tau != None:
        zip_params = zip(current_model.parameters(), target_model.parameters())
        for current_param, target_param in zip_params:
                towards_current = tau*current_param.data + (1.0-tau)*target_param.data
                target_param.data.copy_(towards_current)
    else:
        target_model.load_state_dict(current_model.state_dict())

def plot_array(idx: int, episode: int, array: List, title: str, path: str):

    """Plot different metrics of interest like rewards or losses."""

    plt.figure(figsize=(10,5))
    plt.title(f'{title} - idx {idx} - episode {episode}')
    plt.plot(array)
    plt.grid()
    path_plots = os.path.join(path, 'plots')
    os.makedirs(path_plots, exist_ok=True)
    plt.savefig(f'{path_plots}/{title}.png')
    plt.close()

def save_arrays(arrays_dict: Dict, name: str, output_folder: str):
    
    """Save arrays of interest like rewards or losses."""

    path_csvs = os.path.join(output_folder, 'csvs')
    os.makedirs(path_csvs, exist_ok=True)
    df = pd.DataFrame(arrays_dict)
    df.to_csv(os.path.join(path_csvs, f'{name}.csv'))
