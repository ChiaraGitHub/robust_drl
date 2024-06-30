# General imports
import math
import torch
import numpy as np
import torch.nn as nn

# From other scripts
from src_ppo.torch_utils import determinant


################################################################################
# WEIGHTS INITIALIZATION
# initialize_weights
################################################################################


def orthogonal_init(tensor, gain=1):
    '''
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, initialization_type, scale=2**0.5):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")


################################################################################
# VALUE NETWORK
# Generic Value network MLP
################################################################################


class ValueDenseNet(nn.Module):
    '''
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 64-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    '''
    def __init__(self, state_dim, init=None, hidden_sizes=(64, 64), activation=None):
        '''
        Initializes the value network.
        Inputs:
        - state_dim, the input dimension of the network (i.e dimension of state)
        - hidden_sizes, an iterable of integers, each of which represents the size
        of a hidden layer in the neural network.
        Returns: Initialized Value network
        '''
        super().__init__()
        if isinstance(activation, str):
            self.activation = activation_with_name(activation)()
        else:
            # Default to tanh.
            self.activation = nn.Tanh()
        self.affine_layers = nn.ModuleList()

        prev = state_dim
        for h in hidden_sizes:
            l = nn.Linear(prev, h)
            if init is not None:
                initialize_weights(l, init)
            self.affine_layers.append(l)
            prev = h

        self.final = nn.Linear(prev, 1)
        if init is not None:
            initialize_weights(self.final, init, scale=1.0)

    def initialize(self, init="orthogonal"):
        for l in self.affine_layers:
            initialize_weights(l, init)
        initialize_weights(self.final, init, scale=1.0)

    def forward(self, x):
        '''
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        '''
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        value = self.final(x)
        return value

    def get_value(self, x):
        return self(x)


###############################################################################
# POLICY NETWORKS
# Discrete and Continuous Policy Examples
###############################################################################

'''
Policy network needs to have as inputs state_dim and action_dim.
Must provide:
- A __call__ / forward override (for nn.Module): 
    It returns a tensor parameterizing the ACTIONS DISTRIBUTIONS, given as input
    a tensor of shape (BATCH_SIZE x state_dim).
- A function calc_kl(p, q): 
    It takes two batches of tensors which parameterize probability 
    distributions (of the same form as the output from forward), 
    and returns the KL-DIVERGENCE KL(p||q) tensor of length BATCH_SIZE.
- A function entropies(p):
    It takes in a batch of tensors parameterizing distributions (of the same 
    form as the output from forward), and returns the ENTROPY of each element 
    in the batch as a tensor.
- A function sample(p): 
    It takes in a batch of tensors parameterizing distributions (of the same 
    form as the output from forward) and returns a batch of ACTIONS to be 
    performed.
- A function get_likelihoods(p, actions):
    It takes in a batch of parameterizing tensors (of the same 
    form as the output from forward) and an equal-length batch of actions, 
    and returns a batch of probabilities indicating how likely each action 
    was according to p.
'''


class DiscretePolicy(nn.Module):
    '''
    A discrete policy using a fully connected neural network.
    The parameterizing tensor is a categorical distribution over actions.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=(64, 64)):
        '''
        Initializes the network with the state dimensionality and # actions
        Inputs:
        - state_dim, dimensionality of the state vector
        - action_dim, # of possible discrete actions
        - hidden_sizes, an iterable of length #layers,
            hidden_sizes[i] = number of neurons in layer i
        '''
        super().__init__()
        self.activation = nn.Tanh()

        self.discrete = True
        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.final = nn.Linear(prev_size, action_dim)

    def forward(self, x):
        '''
        Outputs the categorical distribution (via softmax)
        by feeding the state through the neural network
        '''

        for affine in self.affine_layers:
            x = self.activation(affine(x))
        
        probs = nn.functional.softmax(self.final(x))
        return probs

    def calc_kl(self, p, q, get_mean=True): # TODO: does not return a list
        '''
        Calculates E KL(p||q):
        E[sum p(x) log(p(x)/q(x))]
        Inputs:
        - p, first probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        - q, second probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        Returns:
        - Empirical KL from p to q
        '''
        p, q = p.squeeze(), q.squeeze()

        kl = (p * (torch.log(p) - torch.log(q))).sum(-1)
        return kl

    def entropies(self, p):
        '''
        p is probs of shape (batch_size, action_space). return mean entropy
        across the batch of states
        '''
        entropies = (p * torch.log(p)).sum(dim=1)
        return entropies

    def get_loglikelihood(self, p, actions):
        '''
        Inputs:
        - p, batch of probability tensors
        - actions, the actions taken
        '''
        try:
            dist = torch.distributions.categorical.Categorical(p)
            return dist.log_prob(actions)
        except Exception as e:
            raise ValueError("Numerical error")
    
    def sample(self, probs):
        '''
        given probs, return: actions sampled from P(.|s_i), and their
        probabilities
        - s: (batch_size, state_dim)
        Returns actions:
        - actions: shape (batch_size,)
        '''
        dist = torch.distributions.categorical.Categorical(probs)
        actions = dist.sample()
        return actions.long()


class ContinuousPolicy(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=(64, 64),
                 activation=None, use_merged_bias=False):
        super().__init__()
        if isinstance(activation, str):
            self.activation = activation_with_name(activation)()
        else:
            # Default to tanh.
            self.activation = nn.Tanh()
        print('Using activation function', self.activation)
        self.action_dim = action_dim
        self.discrete = False
        self.use_merged_bias = use_merged_bias

        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            if use_merged_bias:
                # Use an extra dimension for weight perturbation, simulating bias.
                lin = nn.Linear(prev_size + 1, i, bias=False)
            else:
                lin = nn.Linear(prev_size, i, bias=True)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        if use_merged_bias:
            self.final_mean = nn.Linear(prev_size + 1, action_dim, bias=False)
        else:
            self.final_mean = nn.Linear(prev_size, action_dim, bias=True)
        initialize_weights(self.final_mean, init, scale=0.01)
        


        stdev_init = torch.zeros(action_dim)
        self.log_stdev = torch.nn.Parameter(stdev_init)

    def forward(self, x):
        # If the time is in the state, discard it
        for affine in self.affine_layers:
            if self.use_merged_bias:
                # Generate an extra "one" for each element, which acts as a bias.
                bias_padding = torch.ones(x.size(0),1)
                x = torch.cat((x, bias_padding), dim=1)
            else:
                pass
            x = self.activation(affine(x))
        
        if self.use_merged_bias:
            bias_padding = torch.ones(x.size(0),1)
            x = torch.cat((x, bias_padding), dim=1)
        means = self.final_mean(x)
        std = torch.exp(self.log_stdev)

        return means, std 

    def sample(self, p):
        '''
        Given prob dist (mean, var), return: actions sampled from p_i, and their
        probabilities. p is tuple (means, var). means shape 
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        means, std = p
        return (means + torch.randn_like(means)*std).detach()

    def get_loglikelihood(self, p, actions):
        try:    
            mean, std = p
            nll =  0.5 * ((actions - mean) / std).pow(2).sum(-1) \
                   + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
                   + self.log_stdev.sum(-1)
            return -nll
        except Exception as e:
            raise ValueError("Numerical error")

    def calc_kl(self, p, q):
        '''
        Get the expected KL distance between two sets of gaussians over states -
        gaussians p and q where p and q are each tuples (mean, var)
        - In other words calculates E KL(p||q): E[sum p(x) log(p(x)/q(x))]
        - From https://stats.stackexchange.com/a/60699
        '''
        p_mean, p_std = p
        q_mean, q_std = q
        p_var, q_var = p_std.pow(2), q_std.pow(2)
        

        d = q_mean.shape[1]
        diff = q_mean - p_mean

        log_quot_frac = torch.log(q_var).sum() - torch.log(p_var).sum()
        tr = (p_var / q_var).sum()
        quadratic = ((diff / q_var) * diff).sum(dim=1)

        kl_sum = 0.5 * (log_quot_frac - d + tr + quadratic)
        assert kl_sum.shape == (p_mean.shape[0],)
        return kl_sum

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (mean, var), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        _, std = p
        detp = determinant(std)
        d = std.shape[0]
        entropies = torch.log(detp) + .5 * (d * (1. + math.log(2 * math.pi)))
        return entropies


###############################################################################
# Policy and value networks from dict and activations from dict
###############################################################################
    
POLICY_NETS = {
    "DiscretePolicy": DiscretePolicy,
    "ContinuousPolicy": ContinuousPolicy,
}

VALUE_NETS = {
    "ValueNet": ValueDenseNet,
}

ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "hardtanh": nn.Hardtanh,
}

def activation_with_name(name):
    return ACTIVATIONS[name]

def policy_net_with_name(name):
    return POLICY_NETS[name]

def value_net_with_name(name):
    return VALUE_NETS[name]