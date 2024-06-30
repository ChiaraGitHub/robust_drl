import torch
from torch import autograd

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
device = "cuda" if torch.cuda.is_available() else "cpu"

def pgd(model, state, action, params={}):

    """Projected Gradient Descent Attack
    """
    
    # Parameters ---------------------------------------------------------------
    epsilon = params['epsilon'] # Max entity of perturbation (% of range)
    n_iters = params['n_steps'] # Number of PGD steps
    state_min = torch.Tensor(params['state_min']).to(device) # Min of each state
    state_max = torch.Tensor(params['state_max']).to(device) # Max of each state
    loss_func = params['loss_func'] # Loss function

    # Get the perturbation sphere ----------------------------------------------
    # Max perturbation in range (e.g. 0.03)
    range_percent = epsilon * (state_max-state_min) 
    # Limit min value perturbation (e.g. -0.015)
    sphere_min = range_percent * (torch.zeros(state.data.size()).to(device) - 0.5)
    # Limit max value perturbation (e.g. +0.015)
    sphere_max = range_percent * (torch.ones(state.data.size()).to(device) - 0.5)
    # Step size, absolute and centered in 0 so just positive limit (e.g. 0.015/10)
    step_size = sphere_max / n_iters 
    action = Variable(torch.tensor(action))
    
    # Define starting point ----------------------------------------------------
    # If random start, start from state + noise
    if params['random_start']:
        # Noise is an epsilon percentage of the range
        noise = range_percent * (torch.rand(state.data.size()).to(device) - 0.5)
        # Clamp the state + noise
        state_adv = torch.clamp(state.data + noise, state_min, state_max)
        state_adv = Variable(state_adv.data, requires_grad=True)
    # Otherwise start from state
    else:
        state_adv = Variable(state.data, requires_grad=True)
    
    # Iterate to add adversarial perturbation based on PGD ---------------------
    for i in range(n_iters):
        
        # Get action logits from adv state
        action_logits = model.forward(state_adv)
        # Compute the loss in terms of action change
        loss = loss_func(action_logits, action) 
        # Train
        model.features.zero_grad()
        loss.backward()
        # Take a step in the direction of the gradient ascent - sign only 
        # (e.g. +- 0.0015)
        eta = step_size * state_adv.grad.data.sign()
        # Modify the adversarial state adding eta (e.g. state + 0.0015)
        state_adv = Variable(state_adv.data + eta, requires_grad=True)
        # Adjust to be within [sphere_min, sphere_max] away from the state
        eta = torch.clamp(state_adv.data - state.data, sphere_min, sphere_max)
        # Finally add the adapted eta to the state
        state_adv.data = state.data + eta
        # Clamp to acceptable state limits and check
        state_adv.data = torch.clamp(state_adv.data, state_min, state_max)
    
    return state_adv.data


def attack(model, state, attack_config, loss_func=torch.nn.CrossEntropyLoss()):
    
    # Get relevant parameters
    params = attack_config['params']
    params['loss_func'] = loss_func
    
    # Act from current state
    action = model.act(state)

    # PGD adversarial state given the state and action
    state_adv = pgd(model, state, action, params=params)

    return state_adv

