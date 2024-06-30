# General import
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # No display

# From other script
from src_ppo.convex_relaxation import get_state_kl_bound


def surrogate_reward(advantages, log_ps_new, log_ps_old, clip_eps=None):
    '''
    Computes the surrogate reward for TRPO and PPO:
    R(\theta) = E[r_t * A_t]
    with support for clamping the ratio (for PPO), s.t.
    R(\theta) = E[clamp(r_t, 1-e, 1+e) * A_t]
    Inputs:
    - adv, unnormalized advantages as calculated by the agents
    - log_ps_new, the log probabilities assigned to taken events by \theta_{new}
    - log_ps_old, the log probabilities assigned to taken events by \theta_{old}
    - clip_EPS, the clipping boundary for PPO loss
    Returns:
    - The surrogate loss as described above
    '''

    # Normalized Advantages
    if len(advantages) == 1:
        n_advs =  advantages
    else:
        std = advantages.std()
        mean = advantages.mean()
        n_advs = (advantages - mean)/(std + 1e-8)

    # Ratio of new probabilities to old ones (in log subtraction)
    ratio_new_old = torch.exp(log_ps_new - log_ps_old)

    # Clamping (for use with PPO)
    if clip_eps is not None:
        ratio_new_old = torch.clamp(ratio_new_old,
                                    1 - clip_eps,
                                    1 + clip_eps)
    return ratio_new_old * n_advs

##############################################################################
# Loss function for the value network
##############################################################################

def value_loss_gae(values, advantages, not_dones, old_values):
    '''
    GAE-based loss for the value function:
        L_t = ((v_t + A_t).detach() - v_{t})
    Optionally, we clip the value function around the original value of v_t

    Inputs: rewards, returns, not_dones, params (from value_step)
    Outputs: value function loss
    '''

    # Desired values are old values plus advantage of the action taken.
    # They do not change during the optimization process.
    # We want the current values to be close to them.
    val_targ = (old_values + advantages).detach()

    # Selected part
    selected_part = not_dones.bool()

    # Values loss unclipped and clipped
    val_loss_mat = (values - val_targ)[selected_part].pow(2)

    # Mean squared error / loss
    mse = val_loss_mat.mean()

    return mse

##############################################################################
# Function for updating the value network
##############################################################################

def value_step(all_states, returns, advantages, not_dones, network,
               val_opt, params, old_values):
    '''
    Update the value function parameterized by a NN
    Inputs:
    - all_states, the states at each timestep
    - returns, discounted rewards (ret_t = r_t + gamma*ret_{t+1})
    - advantages, estimated by GAE
    - not_dones, N * T array with 0s at final steps and 1s everywhere else
    - net, the neural network representing the value function 
    - val_opt, the optimizer for net
    - params, dictionary of parameters
    Returns:
    - Loss of the value regression problem
    '''

    for _ in range(params.VALUE_EPOCHS):
        
        # Create minibatches with shuffling
        state_indices = np.arange(len(returns))
        np.random.shuffle(state_indices)
        splits = np.array_split(state_indices, params.NUM_MINIBATCHES)

        # Update steps
        for split in splits:

            # Filter needed portion
            batch_advantages = advantages[split]
            batch_not_dones = not_dones[split]
            batch_old_values = old_values[split]
            batch_states = all_states[split]
            
            # Zero grad
            val_opt.zero_grad()

            # Forward - value prediction given the states
            values = network(batch_states).squeeze(-1)

            # Value loss
            val_loss = value_loss_gae(values=values,
                                      advantages=batch_advantages,
                                      not_dones=batch_not_dones,
                                      old_values=batch_old_values)

            # Backward propagation
            val_loss.backward()

            # Optimizer step
            val_opt.step()

    return val_loss

##############################################################################
# Function for updating the policy network
##############################################################################

def ppo_step(all_states, actions, old_log_probs,
             advantages, policy_net, params,
             relaxed_net=None, eps_scheduler=None, beta_scheduler=None):
    
    '''
    Proximal Policy Optimization step for both vanilla and robust version
    Inputs:
    - all_states, the historical value of all the states
    - actions, the actions that the policy sampled
    - old_log_probs, the log probability of the actions that the policy sampled
    - advantages, advantages as estimated by GAE
    - policy_net, policy network to train
    - params, additional placeholder for parameters like EPS
    Returns:
    - The PPO loss
    '''

    # If robust
    if relaxed_net is not None:

        # We treat all PPO epochs as one epoch
        eps_scheduler.set_epoch_length(params.PPO_EPOCHS * params.NUM_MINIBATCHES)
        beta_scheduler.set_epoch_length(params.PPO_EPOCHS * params.NUM_MINIBATCHES)
        # We count from 1
        eps_scheduler.step_epoch()
        beta_scheduler.step_epoch()

    # Run number of PPO epochs
    for _ in range(params.PPO_EPOCHS):

        # State shape [experience_steps, observation_size]
        state_indices = np.arange(all_states.shape[0])
        # Shuffle the experience indices
        np.random.shuffle(state_indices)
        # Divide the state indices in NUM_MINIBATCHES splits
        splits = np.array_split(state_indices, params.NUM_MINIBATCHES)

        # Iterate over splits
        for split in splits:

            # Get relevant portion/batch
            batch_states = all_states[split]
            batch_actions = actions[split]
            batch_old_log_ps = old_log_probs[split]
            batch_advantages = advantages[split]

            # Forward propagation, dist contains mean and variance of Gaussian
            distr = policy_net(batch_states) # TODO: check shape

            # From distribution to log likelihood
            new_log_ps = policy_net.get_loglikelihood(distr, batch_actions)

            # Surrogate rewards: exp(new_log_ps - old_log_ps) * advantage
            # shape: minibatch size
            unclp_rew = surrogate_reward(batch_advantages,
                                         log_ps_new=new_log_ps,
                                         log_ps_old=batch_old_log_ps)
            clp_rew = surrogate_reward(batch_advantages,
                                       log_ps_new=new_log_ps,
                                       log_ps_old=batch_old_log_ps,
                                       clip_eps=params.CLIP_EPS)

            # Calculate entropy
            entropy_bonus = policy_net.entropies(distr).mean()

            # If robust
            if relaxed_net is not None:
                # Calculate regularizer under state perturbation.
                eps_scheduler.step_batch()
                beta_scheduler.step_batch()
                batch_action_means = distr[0]
                current_eps = eps_scheduler.get_eps()
                stdev = torch.exp(policy_net.log_stdev)
                
                # TODO: double check
                # Limits calculation
                device = "cuda" if torch.cuda.is_available() else "cpu"
                state_min = torch.Tensor(params.state_min).to(device)
                state_max = torch.Tensor(params.state_max).to(device)
                range_percent = current_eps * (state_max-state_min)
                sphere_min = range_percent * torch.zeros(batch_states.size()) - 0.5 * range_percent
                sphere_max = range_percent * torch.ones(batch_states.size()) - 0.5 * range_percent
                state_ub = torch.clamp(batch_states + sphere_max, max=state_max)
                state_lb = torch.clamp(batch_states + sphere_min, min=state_min)
                kl_upper_bound = get_state_kl_bound(relaxed_net,
                                                    batch_states,
                                                    batch_action_means,
                                                    eps=range_percent*0.5,
                                                    state_lb=None,
                                                    state_ub=None,
                                                    beta=beta_scheduler.get_eps(),
                                                    stdev=stdev).mean()

            # Total loss: mean of min of clipped and unclipped reward for each state + entropy
            surrogate = (-torch.min(unclp_rew, clp_rew)).mean()
            entropy = -params.ENTROPY_COEFF * entropy_bonus
            loss = surrogate + entropy

            # If robust
            if relaxed_net is not None:
                loss = loss + params.KAPPA * kl_upper_bound
                        
            # Backpropagation
            params.POLICY_ADAM.zero_grad()
            loss.backward()
            if params.CLIP_GRAD_NORM != -1:
                torch.nn.utils.clip_grad_norm(policy_net.parameters(),
                                              params.CLIP_GRAD_NORM)
            params.POLICY_ADAM.step()

    return loss.item(), surrogate.item(), entropy.item()


