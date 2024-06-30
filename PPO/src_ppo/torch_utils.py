# General imports
import torch as torch

'''
Common functions/utilities implemented in PyTorch
Sorted into categories:
- General functions
- Actor-critic helpers
- Policy gradient (PPO/TRPO) helpers
- Normalization helpers
- Neural network helpers
- Initialization helpers
'''

########################
### GENERAL UTILITY FUNCTIONS:
# Parameters, unroll, cu_tensorize, cpu_tensorize,
# scat, determinant, safe_op_or_neg_one
########################

class Parameters(dict): 
    og_getattr = dict.__getitem__
    og_setattr = dict.__setitem__

    def __getattr__(self, x):
        try:
            res = self.og_getattr(x.lower()) 
            return res
        except KeyError:
            raise AttributeError(x)

    def __setattr__(self, x, v):
        return self.og_setattr(x.lower(), v)


def unroll(*tensors):
    '''
    Utility function unrolling a list of tensors
    Inputs:
    - tensors; all arguments should be tensors (at least 2D))))
    Returns:
    - The same tensors but with the first two dimensions flattened
    '''
    rets = []
    for t in tensors:
        if t is None:
            rets.append(None)
        else:
            assert len(t.shape) >= 2
            new_shape = [t.shape[0]*t.shape[1]] + list(t.shape[2:])
            rets.append(t.contiguous().view(new_shape))
    return rets


def cu_tensorize(t):
    '''
    Utility function for turning arrays into cuda tensors
    Inputs:
    - t, list
    Returns:
    - Tensor version of t
    '''
    return torch.tensor(t).float().cuda()


def determinant(mat):
    '''
    Returns the determinant of a diagonal matrix
    Inputs:
    - mat, a diagonal matrix
    Returns:
    - The determinant of mat, aka product of the diagonal
    '''
    return torch.exp(torch.log(mat).sum())


########################
### ACTOR-CRITIC HELPERS:
# discount_path, get_path_indices
########################

# Can be used to convert rewards into discounted returns:
# ret[i] = sum of t = i to T of gamma^(t-i) * rew[t]
def discount_path(path, h):
    '''
    Given a "path" of items x_1, x_2, ... x_n, return the discounted
    path, i.e. 
    X_1 = x_1 + h*x_2 + h^2 x_3 + h^3 x_4
    X_2 = x_2 + h*x_3 + h^2 x_4 + h^3 x_5
    etc.
    Can do (more efficiently?) w SciPy. Python here for readability
    Inputs:
    - path, list/tensor of floats
    - h, discount rate
    Outputs:
    - Discounted path, as above
    '''
    curr = 0
    rets = []
    for i in range(len(path)):
        curr = curr*h + path[-1-i]
        rets.append(curr)
    rets =  torch.stack(list(reversed(rets)), 0)
    return rets

def get_path_indices(not_dones):
    
    """
    Returns list of tuples of the form: (time index start, time index end + 1)
    For each path seen in the not_dones array of shape (1, # time steps)
    E.g. if we have not_dones: [[1, 1, 0, 1, 1, 0, 1, 1, 0, 1]] 
    Then we would return: [(3, 5), (5, 9), (9, 10)]
    """
    indices = []
    num_timesteps = not_dones.shape[1]
    last_index = 0
    for i in range(num_timesteps):
        if not_dones[0, i] == 0.:
            indices.append((last_index, i + 1))
            last_index = i + 1
    if last_index != num_timesteps:
        indices.append((last_index, num_timesteps))
    return indices

class Trajectories:
    def __init__(self, states=None, rewards=None, returns=None, not_dones=None,
                 actions=None, action_log_probs=None, advantages=None,
                 unrolled=False, values=None, action_means=None, action_std=None):

        self.states = states
        self.rewards = rewards
        self.returns = returns
        self.values = values
        self.not_dones = not_dones
        self.actions = actions
        self.action_log_probs = action_log_probs
        self.advantages = advantages
        self.action_means = action_means # A batch of vectors.
        self.action_std = action_std # A single vector.
        self.unrolled = unrolled

        """
        # this is disgusting and we should fix it
        if states is not None:
            num_saps = states.shape[0]
            assert states is None or states.shape[0] == num_saps
            assert rewards is None or rewards.shape[0] == num_saps
            assert returns is None or returns.shape[0] == num_saps
            assert values is None or values.shape[0] == num_saps
            assert not_dones is None or not_dones.shape[0] == num_saps
            assert actions is None or actions.shape[0] == num_saps
            assert action_log_probs is None or action_log_probs.shape[0] == num_saps
            assert advantages is None or advantages.shape[0] == num_saps

            self.size = num_saps
        """
            
        
    def unroll(self):
        assert not self.unrolled
        return self.tensor_op(unroll, should_wrap=False)

    def tensor_op(self, lam, should_wrap=True):
        if should_wrap:
            def op(*args):
                return [lam(v) for v in args]
        else:
            op = lam

        tt = op(self.states, self.rewards, self.returns, self.not_dones)
        tt2 = op(self.actions, self.action_log_probs, self.advantages, self.action_means)
        values, = op(self.values)

        ts = Trajectories(states=tt[0], rewards=tt[1], returns=tt[2],
                          not_dones=tt[3], actions=tt2[0],
                          action_log_probs=tt2[1], advantages=tt2[2], action_means=tt2[3], action_std=self.action_std,
                          values=values, unrolled=True)

        return ts
