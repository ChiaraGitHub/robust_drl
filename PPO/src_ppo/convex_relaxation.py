# General imports
import torch
from torch import nn

# From auto_LiRPA
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import *

# From other scipts
from src_ppo.models import activation_with_name


class RelaxedContinuousPolicyForState(nn.Module):

    def __init__(self,
                 activation,
                 policy_model):
        super().__init__()

        self.activation = activation_with_name(activation)()

        print("Create relax model without duplicating parameters")
        # Copy parameters from an existing model, do not create new parameters
        self.affine_layers = policy_model.affine_layers
        # Copy the final mean vector
        self.final_mean = policy_model.final_mean

    '''
    Compute the L2 distance of mean vectors, to bound KL divergence.
    '''
    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        means = self.final_mean(x)
        return means


def get_state_kl_bound(model,
                       states,
                       action_means,
                       eps,
                       state_lb,
                       state_ub,
                       beta=None,
                       stdev=None,
                       use_full_backward=False):
    
    """Compute the bounds of the network."""
    
    # Set each layer's perturbation eps and log_stdev's perturbation
    states = BoundedTensor(states,
                           ptb=PerturbationLpNorm(norm=np.inf, eps=eps, 
                                                  x_L=state_lb, x_U=state_ub)).requires_grad_(False)
    inputs = (states, )

    if use_full_backward:
        # Full backward method, tightest bound.
        ilb, iub = model.compute_bounds(inputs, IBP=False, C=None,
                                        method="backward",
                                        bound_lower=True, bound_upper=True)
        # Fake beta, avoid backward below.
        beta = 1.0
    else:
        # IBP Pass
        ilb, iub = model.compute_bounds(inputs, IBP=True, C=None,
                                        method=None,
                                        bound_lower=True, bound_upper=True)

    # Beta schedule is from 0 to 1
    if 1 - beta < 1e-20:
        lb = ilb
        ub = iub
    else:
        # CROWN Pass
        clb, cub = model.compute_bounds(x=None, IBP=False, C=None,
                                        method='backward',
                                        bound_lower=True, bound_upper=True)
        lb = beta * ilb + (1 - beta) * clb
        ub = beta * iub + (1 - beta) * cub

    lb = lb - action_means
    ub = ub - action_means
    u = torch.max(lb.abs(), ub.abs())

    kl = ((u * u) / (stdev * stdev)).sum(axis=-1, keepdim=True)

    return kl