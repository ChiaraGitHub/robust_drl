import numpy as np

class EpsilonScheduler(object):
    
    def __init__(self, epsilon_start, epsilon_final, duration, start_idx):
        
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.duration = duration
        self.start_idx = start_idx
        self.slope = (self.epsilon_final - self.epsilon_start) / self.duration
    
    def get(self, iter_idx):
        
        # Get the epsilon given the iteration index, with linear scheduling
        if iter_idx < self.start_idx:
            return self.epsilon_start
        else:
            linear_increase = self.epsilon_start + \
                              self.slope * (iter_idx - self.epsilon_start)
            epsi = np.clip(linear_increase,
                           min(self.epsilon_start, self.epsilon_final),
                           max(self.epsilon_start, self.epsilon_final))
            return epsi