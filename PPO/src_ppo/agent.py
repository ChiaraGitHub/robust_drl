# General imports
import os
import sys
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

# From auto_LiRPA
from auto_LiRPA import BoundedModule
from auto_LiRPA.eps_scheduler import LinearScheduler
from auto_LiRPA.bounded_tensor import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# From the other scripts
from src_ppo.models import (value_net_with_name, policy_net_with_name,
                                     ContinuousPolicy, ValueDenseNet)
from src_ppo.torch_utils import (Trajectories,
                                 Parameters,
                                 cu_tensorize,
                                 get_path_indices,
                                 discount_path)
from src_ppo.steps import value_step, ppo_step
from src_ppo.env_setup import Env
from src_ppo.convex_relaxation import RelaxedContinuousPolicyForState


class Trainer():

    """
    Class representing a PPO trainer, which trains both 
    a deep policy network and a deep value network.
    """

    def __init__(self, config, logger, writer):
        
        """
        Initializers a Trainer given all configuration and a logger for 
        logging relevant information.
        """

        # From config to parameters
        self.params = Parameters(config)
        self.logger = logger
        self.writer = writer

        # Use GPU
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Environment initialization
        self.env = Env(game=self.ENV_ID)
        self.params.AGENT_TYPE = "discrete" if self.env.is_discrete else "continuous"
        self.params.NUM_ACTIONS = self.env.num_actions
        self.params.NUM_FEATURES = self.env.num_features
        
        # Define the type of steps for the policy and adversary policy
        self.policy_step = ppo_step # From steps
        self.adversary_policy_step = ppo_step if self.MODE == 'adv_ppo' else None
        
        self.n_steps = 0 # Counter for n_steps

        # Models / optimizers / schedulers instantiation -----------------------

        # Get agent policy network (actor) and value network (critic)
        policy_net_class = policy_net_with_name(config['policy_net_type'])
        value_net_class = value_net_with_name(config['value_net_type'])
        self.policy_net_class = policy_net_class
        self.value_net_class = value_net_class

        # Instantiation policy network
        self.policy_model = self.policy_net_class(
                                            self.NUM_FEATURES,
                                            self.NUM_ACTIONS,
                                            self.INITIALIZATION,
                                            activation=self.policy_activation)

        # Adam optimizer for policy model
        self.params.POLICY_ADAM = optim.Adam(self.policy_model.parameters(),
                                             lr=self.PPO_LR,
                                             eps=self.ADAM_EPS)

        # Instantiate value network
        self.val_model = self.value_net_class(self.NUM_FEATURES,
                                              self.INITIALIZATION)
        
        # Adam value function optimizer
        self.val_opt = optim.Adam(self.val_model.parameters(),
                                  lr=self.VAL_LR,
                                  eps=self.ADAM_EPS)
        assert self.policy_model.discrete == (self.AGENT_TYPE == "discrete")

        # Learning rate annealing for policy and value function
        if self.ANNEAL_LR:
            lam = lambda f: 1-f/self.TRAIN_ITERATIONS
            ps = optim.lr_scheduler.LambdaLR(self.POLICY_ADAM, 
                                             lr_lambda=lam)
            vs = optim.lr_scheduler.LambdaLR(self.val_opt,
                                             lr_lambda=lam)
            self.params.POLICY_SCHEDULER = ps
            self.params.VALUE_SCHEDULER = vs

        # Instantiate convex relaxation model when mode is 'robust_ppo'
        if self.MODE == 'robust_ppo':
            self.create_relaxed_model_continuous()

        # Policy adversary: here the network as states and inputs and outputs
        # From state to perturbation
        if self.MODE == 'adv_ppo':
            
            # Adversary policy model (here the output is a state)
            self.adversary_policy_model = self.policy_net_class(
                                                self.NUM_FEATURES, # input
                                                self.NUM_FEATURES, # output
                                                self.INITIALIZATION,
                                                activation=self.policy_activation)
            # Optimizer for adversary policy net
            self.params.adversary_poli_opt = optim.Adam(
                                    self.adversary_policy_model.parameters(),
                                    lr=self.PPO_LR,
                                    eps=self.ADAM_EPS) # TODO: put in config

            # Adversary value function model
            self.adversary_val_model = self.value_net_class(self.NUM_FEATURES,
                                                            self.INITIALIZATION)
            
            # Optimizer for adversary value net
            self.adversary_val_opt = optim.Adam(self.adversary_val_model.parameters(),
                                                lr=self.VAL_LR,
                                                eps=self.ADAM_EPS)
            
            # Check discrete/continuous
            assert self.adversary_policy_model.discrete == (self.AGENT_TYPE == "discrete")

            # Learning rate annealling for adversary.
            if self.ANNEAL_LR:
                adv_lam = lambda f: 1 - f/self.TRAIN_ITERATIONS
                adv_ps = optim.lr_scheduler.LambdaLR(self.adversary_poli_opt, 
                                                     lr_lambda=adv_lam)
                adv_vs = optim.lr_scheduler.LambdaLR(self.adversary_val_opt,
                                                     lr_lambda=adv_lam)
                self.params.ADV_POLICY_SCHEDULER = adv_ps
                self.params.ADV_VALUE_SCHEDULER = adv_vs
    
    def __getattr__(self, x):
        """
        Allows accessing self.config_param instead of self.params.config_param
        """
        if x == 'params':
            return {}
        try:
            return getattr(self.params, x)
        except KeyError:
            raise AttributeError(x)
        
    def create_relaxed_model_continuous(self):

        """
        Relaxed model in the case of robust PPO. It is then used
        in the robust ppo step.
        """

        # Implemented only for continuous policy and convex relaxation
        relaxed_policy_model = RelaxedContinuousPolicyForState(
                                            activation=self.policy_activation, # activation
                                            policy_model=self.policy_model) # policy model
        # Create random input with size as the states
        dummy_input1 = torch.randn(1, self.NUM_FEATURES)
        inputs = (dummy_input1, )
        self.relaxed_policy_model = BoundedModule(relaxed_policy_model,
                                                  inputs)
        self.robust_eps_scheduler = LinearScheduler(
                                        self.params.ATTACK_EPS,
                                        self.params.ROBUST_PPO_EPS_SCHEDULER_OPTS)
        self.robust_beta_scheduler = LinearScheduler(
                                        self.params.ROBUST_PPO_BETA,
                                        self.params.ROBUST_PPO_BETA_SCHEDULER_OPTS)
                    
    def advantage_and_return(self, rewards, values, not_dones):
        
        """
        Calculate GAE advantage, discounted returns, and 
        true reward (average reward per trajectory)

        using formula from John Schulman's code:
        V(s_t+1) = {0 if s_t is terminal
                   {v_s_{t+1} if s_t not terminal and t != T (last step)
                   {v_s if s_t not terminal and t == T
        """

        # Concatenate values
        V_s_tp1 = torch.cat([values[:,1:], values[:, -1:]], dim=1) * not_dones

        # GAE: delta_t^V = r_t + discount * V(s_{t+1}) - V(s_t)
        deltas = rewards + self.GAMMA * V_s_tp1 - values

        # Initialize and populate advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        indices = get_path_indices(not_dones) # Indices where not done
        for start, end in indices:
            # Discount each path by gamma * lambda
            advantages[0, start:end] = discount_path(
                                        deltas[0, start:end],
                                        self.LAMBDA*self.GAMMA)
            # Discount by gamma
            returns[0, start:end] = discount_path(rewards[0, start:end],
                                               self.GAMMA)

        return advantages.clone().detach(), returns.clone().detach()

    def step_wrapper(self, action, env): 
        
        # TODO: To simplify/remove
        completed_episode_info = None
        # Step wrapper
        gym_action = action[0].cpu().numpy()
        new_state, normed_reward, is_done, info = env.step(gym_action)
        if is_done:
            completed_episode_info = info['done']
            new_state = env.reset()

        return [completed_episode_info,
                cu_tensorize(normed_reward),
                cu_tensorize(new_state),
                cu_tensorize(int(not is_done))]

    def run_trajectories(self,
                         num_steps,
                         adversary_step=False):
        """
        Resets envs, runs "num_steps" steps. If env reaches terminal state, 
        it is re-started and the length of the episode and reward are return.
        Returns: lengths, rewards, trajectories
        """

        # Reset env and get first state ----------------------------------------
        env = self.env

        # Initialize -----------------------------------------------------------
        # rewards, not dones and action log_probs to zero (1, num_steps)
        rewards = torch.zeros((1, num_steps))
        not_dones = torch.zeros((1, num_steps))
        action_log_probs = torch.zeros((1, num_steps))

        # Get action/output shape (1, num_steps, num_actions)
        if adversary_step:
            assert self.MODE == "adv_ppo"
            # For the adversary, action is a state perturbation!
            actions_shape = (1, num_steps) + (self.NUM_FEATURES,)
        else:
            actions_shape = (1, num_steps) + (self.NUM_ACTIONS,)
        
        # Init action and action means
        actions = torch.zeros(actions_shape)
        action_means = torch.zeros(actions_shape)

        # Init states (1, num_steps, state_shape)
        initial_state = cu_tensorize(env.reset())
        states_shape = (1, num_steps + 1) + (initial_state.shape[0],)
        states =  torch.zeros(states_shape)

        # Applies if (adv_ppo and not adversary_step) or (not adv_ppo)
        # In adversarial training we have an adversarial step and a normal step
        # TRUE if we are not training an adversarial model or
        # if it is an adversarial model but the step is not
        # FALSE if it is adversary step
        collect_perturbed_state = ((self.MODE == "adv_ppo" and not adversary_step) 
                                   or ((not self.MODE == "adv_ppo")))

        if collect_perturbed_state:
            # States are collected AFTER the perturbation
            # We cannot set states[:, 0, :] here as we have not started perturbation yet
            last_states = initial_state.unsqueeze(0)  
        # If it is adversary step save before perturbation
        else:
            # States are collected BEFORE the perturbation
            last_states = initial_state.unsqueeze(0) 
            states[:, 0, :] = initial_state
        
        # Take steps and collect
        completed_episode_info = []
        for t in range(num_steps):

            # Update the last_states with the perturbation
            # Adversarial PPO - both adversarial and not adversarial step
            if self.MODE == "adv_ppo":
                
                # If adversary_step, always apply the optimal attack
                # If agent_step run optimal attack when ADV_ADVERSARY_RATIO >= random.random()
                if adversary_step or self.params.ADV_ADVERSARY_RATIO >= random.random():
                    
                    # Here we use the adversary policy
                    # Get perturbation prob distribution (of states) + means
                    adv_perturb_prob = self.adversary_policy_model(last_states)
                    next_adv_perturb_means, _ = adv_perturb_prob
                    
                    # Sample from the prob distribution
                    next_adv_perturb = self.adversary_policy_model.sample(adv_perturb_prob)
                    
                    # Get log likelihood for this perturbation 
                    # from prob distribution and sample
                    next_adv_perturb_log_probs = self.adversary_policy_model.get_loglikelihood(adv_perturb_prob,
                                                                                               next_adv_perturb)
                    # Add the perturbation to state (we learn a residual)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    state_min = torch.Tensor(self.params.state_min).to(device)
                    state_max = torch.Tensor(self.params.state_max).to(device)
                    range_percent = self.ATTACK_EPS * (state_max-state_min)

                    last_states = last_states + F.hardtanh(next_adv_perturb) * range_percent
                    
                    # The perturbation itself is the action (similar to the next_actions variable below)
                    next_adv_perturb = next_adv_perturb.unsqueeze(1)
            
            # NOT Adversarial PPO
            else:
                # Apply naive adversarial training (not optimal attack)
                maybe_attacked_last_states = self.apply_attack(last_states)
                # For naive adversarial training, the state under perturbation 
                # is used to get the actions. However in the trajectory we still
                # save the state without perturbation as the true env states 
                # are not perturbed.
                # (depending on if self.COLLECT_PERTURBED_STATES is set)

                # Double check if the attack eps is valid
                max_eps = (maybe_attacked_last_states - last_states).abs().max().item()
                if max_eps > float(self.params.ATTACK_EPS) + 1e-5:
                    raise RuntimeError(f"{max_eps} > {float(self.params.ATTACK_EPS)}")
                
                # Last states is set to the attacked one (if even attacked)
                last_states = maybe_attacked_last_states

            # Forward propagation of last_states (perturbed) to get action -----
            action_pds = self.policy_model(last_states)
            next_action_means, next_action_stds = action_pds

            # Sample actions and get log-likelihood
            next_action = self.policy_model.sample(action_pds)
            next_action_log_probs = self.policy_model.get_loglikelihood(action_pds,
                                                                        next_action)
            # TODO: check, there was: .unsqueeze(1)
            next_action_log_probs = next_action_log_probs
            
            
            # Step environment with chosen action ------------------------------
            # done_info has length and total reward for completed trajectories
            (done_info, next_rewards,
             next_states, next_not_dones) = self.step_wrapper(next_action, env)

            # If actor finished AND this is not the last step
            # OR actor finished AND we have no episode information
            # Append to completed_episode_info
            if done_info != None and \
               (t != self.num_steps - 1 or len(completed_episode_info) == 0):
                completed_episode_info.extend(done_info)

            # Update history ---------------------------------------------------
            
            # If it is an ADVERSARIAL STEP
            if adversary_step:
                # Negate the reward for adv training
                # Collect states before perturbation
                next_rewards = -next_rewards
                pairs = [
                    (rewards, next_rewards),
                    (not_dones, next_not_dones),
                    (actions, next_adv_perturb), # State pertubation is here action
                    (action_means, next_adv_perturb_means),
                    (action_log_probs, next_adv_perturb_log_probs),
                    (states, next_states), # Next state without perturbation saved in next position
                ]
            # If it is NOT an ADVERSARIAL STEP
            else:
                # Save the perturbed environment state
                pairs = [
                    (rewards, next_rewards),
                    (not_dones, next_not_dones),
                    (actions, next_action), # Sampled actions
                    (action_means, next_action_means),
                    (action_log_probs, next_action_log_probs),
                    (states, last_states.unsqueeze(1)), # Last perturbed state
                ]

            # Filling the totals (defined above) for the trajectories
            for total_array, new_value in pairs:
                # If adversary step - save at next position the unperturbed state (previous already saved)
                if total_array is states and not collect_perturbed_state: 
                    # In this case we go to index + 1
                    total_array[:, t+1] = new_value
                else:
                    # For normal steps we save perturbed state at current t
                    total_array[:, t] = new_value
            
            # Next step becomes current state
            last_states = next_states.unsqueeze(0)

        # Save LAST STEP (for non adversarial step) - out of loop
        if collect_perturbed_state:
            if self.MODE == "adv_ppo":
                # missing the last state; we have not perturb it yet.
                adv_perturb_prob = self.adversary_policy_model(last_states)
                # sample from the density.
                next_adv_perturb = self.adversary_policy_model.sample(adv_perturb_prob)
                # add the perturbation to state (we learn a residual)
                # Add the perturbation to state (we learn a residual)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                state_min = torch.Tensor(self.params.state_min).to(device)
                state_max = torch.Tensor(self.params.state_max).to(device)
                range_percent = self.ATTACK_EPS * (state_max-state_min)
                last_states = last_states + F.hardtanh(next_adv_perturb) * range_percent
            else:
                last_states = self.apply_attack(last_states)
            states[:, -1] = last_states.unsqueeze(1)


        # Get mean episode length and true rewards over all the trajectories
        infos = np.array(completed_episode_info).reshape(-1,2)
        if infos.size > 0:
            avg_episode_length, avg_episode_reward = np.mean(infos, axis=0)
        else:
            avg_episode_length, avg_episode_reward = np.nan, np.nan

        # Last state is never acted on, discard
        states = states[:,:-1,:]

        # Prepare trajectories
        trajs = Trajectories(rewards=rewards, 
                             action_log_probs=action_log_probs,
                             not_dones=not_dones, 
                             actions=actions,
                             states=states,
                             action_means=action_means,
                             action_std=next_action_stds)

        return avg_episode_length, avg_episode_reward, trajs

    def apply_attack(self, last_state):
        """Apply different types of attack."""

        eps = float(self.params.ATTACK_EPS)
        steps = self.params.ATTACK_STEPS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_min = torch.Tensor(self.params['state_min']).to(device)
        state_max = torch.Tensor(self.params['state_max']).to(device)
        
        if self.params.ATTACK_METHOD == "critic":
            # Find a state that is close last_states and decreases value most
            if steps > 0:
                
                # TODO: check if this version is fine
                range_percent = eps * (state_max-state_min) # e.g. 0.1*3=0.3
                sphere_min = range_percent * (torch.zeros(last_state.size()).to(device) - 0.5) # e.g. -0.15
                sphere_max = range_percent * (torch.ones(last_state.size()).to(device) - 0.5) #e.g. +0.15
                clamp_min = last_state + sphere_min # e.g. 2-0.15=1.85
                clamp_max = last_state + sphere_max # e.g. 2+0.15=2.15
                noise = range_percent/steps * (torch.rand(last_state.size()).to(device) - 0.5) # something between -0.15/steps & 0.15/steps
                states = last_state + noise # e.g. 2+0.1=2.1
                step_eps = sphere_max/steps

                # Old version ---
                # # Total epsilon divided by the steps - step size for one iteration
                # step_eps = eps / steps 
                # # Max acceptable deviation
                # clamp_min = last_state - eps
                # clamp_max = last_state + eps
                # # Random start inside the epsilon range for that step
                # noise = torch.empty_like(last_state).uniform_(-step_eps, step_eps)
                # states = last_state + noise
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        value = self.val_model(states).mean(dim=1)
                        value.backward()
                        update = states.grad.sign() * step_eps
                        # Clamp to +/- eps - "-" to minimise the value
                        states.data = torch.clamp(states.data - update,
                                                  clamp_min,
                                                  clamp_max)
                    self.val_model.zero_grad()
                return states.detach()
            else:
                return last_state
        elif self.params.ATTACK_METHOD == "random":
            # Apply uniform random noise
            # New version
            range_percent = eps * (state_max-state_min) # e.g. 0.1*3=0.3
            sphere_min = range_percent * (torch.zeros(last_state.size()).to(device) - 0.5) # e.g. -0.15
            sphere_max = range_percent * (torch.ones(last_state.size()).to(device) - 0.5) #e.g. +0.15
            clamp_min = last_state + sphere_min # e.g. 2-0.15=1.85
            clamp_max = last_state + sphere_max # e.g. 2+0.15=2.15
            noise = range_percent * (torch.rand(last_state.size()).to(device) - 0.5) # something between -0.15 & +0.15
            # Old version --
            # noise = torch.empty_like(last_state).uniform_(-eps, eps)
            return (last_state + noise).detach()
        elif self.params.ATTACK_METHOD == "action":
            if steps > 0:
                
                # TODO: check if this version is fine
                range_percent = eps * (state_max-state_min) # e.g. 0.1*3=0.3
                sphere_min = range_percent * (torch.zeros(last_state.size()).to(device) - 0.5) # e.g. -0.15
                sphere_max = range_percent * (torch.ones(last_state.size()).to(device) - 0.5) #e.g. +0.15
                clamp_min = last_state + sphere_min # e.g. 2-0.15=1.85
                clamp_max = last_state + sphere_max # e.g. 2+0.15=2.15
                step_eps = sphere_max/steps # only half range (0.15 not 0.3 as with sign)

                # Old version ---
                # # Total epsilon divided by the steps - step size for one iteration
                # step_eps = eps / steps
                # # Max acceptable deviation
                # clamp_min = last_state - eps
                # clamp_max = last_state + eps

                # SGLD noise factor with beta=1
                noise_factor = torch.sqrt(2 * step_eps)
                noise = torch.randn_like(last_state) * noise_factor
                # The first step has gradient zero, so add the noise and projection directly.
                states = last_state + noise.sign() * step_eps
                # Current action at this state.
                old_action, old_stdev = self.policy_model(last_state)
                # Normalize stdev, avoid numerical issue
                old_stdev /= (old_stdev.mean())
                old_action = old_action.detach()
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        action_change = (self.policy_model(states)[0] - old_action) / old_stdev
                        action_change = (action_change * action_change).sum(dim=1)
                        action_change.backward()
                        # Reduce noise at every step
                        noise_factor = torch.sqrt(2 * step_eps) / (i+2)
                        # Project noisy gradient to step boundary
                        update = (states.grad + noise_factor * torch.randn_like(last_state)).sign() * step_eps
                        # Clamp to +/- eps - "+" to maximize action change
                        states.data = torch.min(torch.max(states.data + update, clamp_min), clamp_max)
                     
                    self.policy_model.zero_grad()
                return states.detach()
            else:
                return last_state
        elif self.params.ATTACK_METHOD == "action_v2":
            if steps > 0:

                # NOTE the difference is in the noise factor (/ not *)

                # TODO: check if this version is fine
                range_percent = eps * (state_max-state_min) # e.g. 0.1*3=0.3
                sphere_min = range_percent * (torch.zeros(last_state.size()).to(device) - 0.5) # e.g. -0.15
                sphere_max = range_percent * (torch.ones(last_state.size()).to(device) - 0.5) #e.g. +0.15
                clamp_min = last_state + sphere_min # e.g. 2-0.15=1.85
                clamp_max = last_state + sphere_max # e.g. 2+0.15=2.15
                step_eps = sphere_max/steps # only half range (0.15 not 0.3 as with sign)

                # Old version ---
                # # Total epsilon divided by the steps - step size for one iteration
                # step_eps = eps / steps
                # # Max acceptable deviation
                # clamp_min = last_state - eps
                # clamp_max = last_state + eps

                # SGLD noise factor with beta=1
                noise_factor = np.sqrt(2 / step_eps)
                noise = torch.randn_like(last_state) * noise_factor
                # The first step has gradient zero, so add the noise and projection directly.
                states = last_state + noise.sign() * step_eps
                # Current action at this state.
                old_action, old_stdev = self.policy_model(last_state)
                # Normalize stdev, avoid numerical issue
                old_stdev /= (old_stdev.mean())
                old_action = old_action.detach()
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        action_change = (self.policy_model(states)[0] - old_action) / old_stdev
                        action_change = (action_change * action_change).sum(dim=1)
                        action_change.backward()
                        # Reduce noise at every step
                        noise_factor = np.sqrt(2 / step_eps) / (i+2)
                        # Project noisy gradient to step boundary
                        update = (states.grad + noise_factor * torch.randn_like(last_state)).sign() * step_eps
                        # Clamp to +/- eps - "+" to maximize action change
                        states.data = torch.min(torch.max(states.data + update, clamp_min), clamp_max)
                     
                    self.policy_model.zero_grad()
                return states.detach()
            else:
                return last_state
        elif self.params.ATTACK_METHOD == "sarsa" or self.params.ATTACK_METHOD == "sarsa+action":
            # Attack using a learned value network.
            assert self.params.sarsa_model_path is not None
            use_action = self.params.ATTACK_SARSA_ACTION_RATIO > 0 and self.params.ATTACK_METHOD == "sarsa+action"
            action_ratio = self.params.ATTACK_SARSA_ACTION_RATIO
            assert action_ratio >= 0 and action_ratio <= 1
            if not hasattr(self, "sarsa_network"):
                self.sarsa_network = ValueDenseNet(state_dim=self.NUM_FEATURES+self.NUM_ACTIONS, init="normal")
                print("Loading sarsa network", self.params.sarsa_model_path)
                sarsa_ckpt = torch.load(os.path.join(self.params.base_test_path,
                                                     self.params.sarsa_model_path))
                sarsa_meta = sarsa_ckpt['metadata']
                sarsa_eps = sarsa_meta['sarsa_eps'] if 'sarsa_eps' in sarsa_meta else "unknown"
                sarsa_reg = sarsa_meta['sarsa_reg'] if 'sarsa_reg' in sarsa_meta else "unknown"
                sarsa_steps = sarsa_meta['sarsa_steps'] if 'sarsa_steps' in sarsa_meta else "unknown"
                print(f"Sarsa network was trained with eps={sarsa_eps}, reg={sarsa_reg}, steps={sarsa_steps}")
                if use_action:
                    print(f"objective: {1.0 - action_ratio} * sarsa + {action_ratio} * action_change")
                else:
                    print("Not adding action change objective.")
                self.sarsa_network.load_state_dict(sarsa_ckpt['state_dict'])
            if steps > 0:

                # TODO: check if this version is fine
                range_percent = eps * (state_max-state_min) # e.g. 0.1*3=0.3
                sphere_min = range_percent * (torch.zeros(last_state.size()).to(device) - 0.5) # e.g. -0.15
                sphere_max = range_percent * (torch.ones(last_state.size()).to(device) - 0.5) #e.g. +0.15
                clamp_min = last_state + sphere_min # e.g. 2-0.15=1.85
                clamp_max = last_state + sphere_max # e.g. 2+0.15=2.15
                noise = range_percent/steps * (torch.rand(last_state.size()).to(device) - 0.5) # something between -0.15/steps & 0.15/steps
                states = last_state + noise # e.g. 2+0.1=2.1
                step_eps = sphere_max/steps # only half range (0.15 not 0.3 as with sign)


                # Old version ---
                # step_eps = eps / steps

                # clamp_min = last_state - eps
                # clamp_max = last_state + eps
                # Random start.
                # noise = torch.empty_like(last_state).uniform_(-step_eps, step_eps)
                # states = last_state + noise
                if use_action:
                    # Current action at this state.
                    old_action, old_stdev = self.policy_model(last_state)
                    old_stdev /= (old_stdev.mean())
                    old_action = old_action.detach()
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        # This is the mean action...
                        actions = self.policy_model(states)[0]
                        value = self.sarsa_network(torch.cat((last_state, actions), dim=1)).mean(dim=1)
                        if use_action:
                            action_change = (actions - old_action) / old_stdev
                            # We want to maximize the action change, thus the minus sign.
                            action_change = -(action_change * action_change).mean(dim=1)
                            loss = action_ratio * action_change + (1.0 - action_ratio) * value
                        else:
                            action_change = 0.0
                            loss = value
                        loss.backward()
                        update = states.grad.sign() * step_eps
                        # Clamp to +/- eps - "-" to minimise the value
                        states.data = torch.clamp(states.data - update,
                                                  clamp_min,
                                                  clamp_max)
                    self.val_model.zero_grad()
                return states.detach()
            else:
                return last_state
        elif self.params.ATTACK_METHOD == "none":
            return last_state
        else:
            raise ValueError(f'Unknown attack method {self.params.ATTACK_METHOD}')

    def collect_state_action_pairs(self,
                                   num_steps,
                                   test=False,
                                   adversary_step=False):
        
        """Run trajectories and return state-action pairs (SAPs)
           including values, advantages and returns for each state.
           The information about wheather or not it is an adversary step 
           is used to select the VALUE network for forward propagation."""

        with torch.no_grad():
            
            # Run trajectories, get avg length, avg reward, trajectories
            (avg_ep_length,
             avg_ep_reward,
             trajs) = self.run_trajectories(
                                    num_steps,
                                    adversary_step=adversary_step)

            # If not testing compute values / advantages / returns
            # In testing no need to compute values and advantage
            if not test:

                # VALUES
                if adversary_step:
                    # Forward ADVERSARY VALUE function passing traj states
                    # values size = [1, num_saps]
                    values = self.adversary_val_model(trajs.states).squeeze(-1)
                else:
                    # Forward NORMAL VALUE function passing traj states 
                    # values size = [1, num_saps]
                    values = self.val_model(trajs.states).squeeze(-1)

                # ADVANTAGES & RETURN
                # returns (size = [1, num_saps])
                advantages, returns = self.advantage_and_return(
                                                    rewards=trajs.rewards,
                                                    values=values,
                                                    not_dones=trajs.not_dones)
                # Add values, advantages and returns to trajectories
                trajs.values = values
                trajs.advantages = advantages
                trajs.returns = returns
                
            # TODO: maybe change
            # All vectors are squeezed
            state_action_pairs = trajs.unroll()

        return state_action_pairs, avg_ep_reward, avg_ep_length

    def value_and_policy_update_steps(self,
                   state_action_pairs,
                   adversary_step=False):
        
        # Set up models and params to be considered in the normal / adversary case
        # ----------------------------------------------------------------------
        if adversary_step:
            policy_model = self.adversary_policy_model
            if self.ANNEAL_LR:
                policy_scheduler = self.ADV_POLICY_SCHEDULER
                val_scheduler = self.ADV_VALUE_SCHEDULER
            val_model = self.adversary_val_model
            val_opt = self.adversary_val_opt

            # Some parameters are overwritten in the adversary case
            policy_params = Parameters(self.params.copy())
            policy_params.POLICY_ADAM = self.adversary_poli_opt
            
        else:
            policy_model = self.policy_model
            if self.ANNEAL_LR:
                policy_scheduler = self.POLICY_SCHEDULER
                val_scheduler = self.VALUE_SCHEDULER
            val_model = self.val_model
            val_opt = self.val_opt

            policy_params = self.params

        # BACKWARD PROPAGATION (it is in the step function)
        # Update the value function --------------------------------------------
        val_loss = value_step(all_states=state_action_pairs.states,
                              returns=state_action_pairs.returns,
                              advantages=state_action_pairs.advantages,
                              not_dones=state_action_pairs.not_dones,
                              network=val_model,
                              val_opt=val_opt,
                              params=self.params,
                              old_values=state_action_pairs.values.detach())
        val_loss = val_loss.mean()

        # If anneal_lr decrease the value function learning rate ---------------
        if self.ANNEAL_LR:
            val_scheduler.step()

        # Update the policy network --------------------------------------------

        # Prepare arguments for policy network
        args = [state_action_pairs.states,
                state_action_pairs.actions,
                state_action_pairs.action_log_probs,
                state_action_pairs.advantages,
                policy_model,
                policy_params] 

        # Add arguments for robust version
        if (self.MODE == 'robust_ppo') and \
            isinstance(self.policy_model, ContinuousPolicy) and \
            not adversary_step:
            
            args += [self.relaxed_policy_model,
                     self.robust_eps_scheduler,
                     self.robust_beta_scheduler]

        # BACKWARD PROPAGATION (it is in the step function)
        # Policy optimization step
        # return policy loss surr loss and entropy bonus
        if adversary_step:
            policy_loss, surr_loss, entropy_bonus = self.adversary_policy_step(*args)
        else:
            policy_loss, surr_loss, entropy_bonus = self.policy_step(*args)

        # If anneal_lr decrease the policy network learning rate ---------------
        if self.ANNEAL_LR:
            policy_scheduler.step()

        val_loss = val_loss.mean().item()
        return policy_loss, surr_loss, entropy_bonus, val_loss

    def train_step_entrypoint(self):
        
        # ADVERSARIAL step
        if self.MODE == "adv_ppo":

            # Take one POLICY step
            # Collect trajectories + backward propagation
            avg_ep_reward, avg_ep_length = self.train_step_implementation(
                                                    adversary_step=False)
            # Take one ADVERSARY step to train the adversary (for attack)
            # Collect trajectories + backward propagation
            self.train_step_implementation(adversary_step=True)
        
        # NORMAL step
        else:
            # Collect trajectories + backward
            avg_ep_reward, avg_ep_length = self.train_step_implementation(
                                                    adversary_step=False)

        # Increase step counter
        self.n_steps += 1

        return avg_ep_reward, avg_ep_length

    def train_step_implementation(self,
                                  adversary_step=False):
        '''
        Training by: collecting trajectories, calculating advantages
        taking a policy gradient step, taking a value function step
        Returns: current average reward / average episode length
        '''

        # Collect state-action pairs needed for the update below
        (state_action_pairs,
         avg_ep_reward, 
         avg_ep_length) = self.collect_state_action_pairs(
                                        num_steps=self.num_steps,
                                        adversary_step=adversary_step)
        
        # Training - Update value network and policy networks
        (policy_loss, surr_loss,
         entropy_bonus, val_loss) = self.value_and_policy_update_steps(
                                        state_action_pairs=state_action_pairs,
                                        adversary_step=adversary_step)
        # Logging of training status
        self.logger.log(f"- avg_ep_reward: {avg_ep_reward}")
        self.logger.log(f"- avg_ep_length: {avg_ep_length}")
        self.logger.log(f"- policy_loss: {policy_loss}")
        self.logger.log(f"- surr_loss: {surr_loss}")
        self.logger.log(f"- entropy_bonus: {entropy_bonus}")
        self.logger.log(f"- val_loss: {val_loss}")

        # TensorBoard
        if self.writer:
            self.writer.add_scalar("Plots/avg_ep_reward", avg_ep_reward, self.n_steps)
            self.writer.add_scalar("Plots/avg_ep_length", avg_ep_length, self.n_steps)
            self.writer.add_scalar("Plots/policy_loss", policy_loss, self.n_steps)
            self.writer.add_scalar("Plots/val_loss", val_loss, self.n_steps)
    
        return avg_ep_reward, avg_ep_length

    def sarsa_setup(self, lr_schedule, eps_scheduler, beta_scheduler):
        
        # Create the Sarsa model, with S and A as the input
        self.sarsa_model = ValueDenseNet(self.NUM_FEATURES + self.NUM_ACTIONS,
                                         self.INITIALIZATION)
        self.sarsa_opt = optim.Adam(self.sarsa_model.parameters(),
                                    lr=self.VAL_LR, eps=self.ADAM_EPS)
        self.sarsa_scheduler = optim.lr_scheduler.LambdaLR(self.sarsa_opt,
                                                           lr_schedule)
        self.sarsa_eps_scheduler = eps_scheduler
        self.sarsa_beta_scheduler = beta_scheduler
        
        # Convert model with relaxation wrapper
        dummy_input = torch.randn(1, self.NUM_FEATURES + self.NUM_ACTIONS)
        self.relaxed_sarsa_model = BoundedModule(self.sarsa_model, dummy_input)
    
    def sarsa_steps(self, saps):
        # Begin advanged logging code
        assert saps.unrolled
        loss = torch.nn.SmoothL1Loss()
        action_std = torch.exp(self.policy_model.log_stdev).detach().requires_grad_(False)  # Avoid backprop twice.
        # We treat all value epochs as one epoch.
        self.sarsa_eps_scheduler.set_epoch_length(self.params.VALUE_EPOCHS * self.params.NUM_MINIBATCHES)
        self.sarsa_beta_scheduler.set_epoch_length(self.params.VALUE_EPOCHS * self.params.NUM_MINIBATCHES)
        # We count from 1.
        self.sarsa_eps_scheduler.step_epoch()
        self.sarsa_beta_scheduler.step_epoch()
        # saps contains state->action->reward and not_done.
        for i in range(self.params.VALUE_EPOCHS):
            # Create minibatches with shuffuling
            state_indices = np.arange(saps.rewards.nelement())
            np.random.shuffle(state_indices)
            splits = np.array_split(state_indices, self.params.NUM_MINIBATCHES)

            # Minibatch SGD
            for selected in splits:
                def sel(*args):
                    return [v[selected] for v in args]

                self.sarsa_opt.zero_grad()
                sel_states, sel_actions, sel_rewards, sel_not_dones = sel(saps.states, saps.actions, saps.rewards, saps.not_dones)
                self.sarsa_eps_scheduler.step_batch()
                self.sarsa_beta_scheduler.step_batch()
                
                inputs = torch.cat((sel_states, sel_actions), dim=1)
                # action_diff = self.sarsa_eps_scheduler.get_eps() * action_std
                # inputs_lb = torch.cat((sel_states, sel_actions - action_diff), dim=1).detach().requires_grad_(False)
                # inputs_ub = torch.cat((sel_states, sel_actions + action_diff), dim=1).detach().requires_grad_(False)
                # bounded_inputs = BoundedTensor(inputs, ptb=PerturbationLpNorm(norm=np.inf, eps=None, x_L=inputs_lb, x_U=inputs_ub))
                current_eps = self.sarsa_eps_scheduler.get_eps()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                state_min = torch.Tensor(self.params.state_min).to(device)
                state_max = torch.Tensor(self.params.state_max).to(device)
                range_percent = current_eps * (state_max-state_min)
                # sphere_min = range_percent * torch.zeros(sel_states.size()) - 0.5 * range_percent
                # sphere_max = range_percent * torch.ones(sel_states.size()) - 0.5 * range_percent
                # state_ub = torch.clamp(sel_states + sphere_max, max=state_max)
                # state_lb = torch.clamp(sel_states + sphere_min, min=state_min)

                eps = torch.cat((range_percent*0.5,
                                 torch.tensor([current_eps, current_eps])))
                bounded_inputs = BoundedTensor(inputs, ptb=PerturbationLpNorm(norm=np.inf, eps=eps))

                q = self.relaxed_sarsa_model(bounded_inputs).squeeze(-1)
                q_old = q[:-1]
                q_next = q[1:] * self.GAMMA * sel_not_dones[:-1] + sel_rewards[:-1]
                q_next = q_next.detach()
                # q_loss = (q_old - q_next).pow(2).sum(dim=-1).mean()
                q_loss = loss(q_old, q_next)
                # Compute the robustness regularization.
                if self.sarsa_eps_scheduler.get_eps() > 0 and self.params.SARSA_REG > 0:
                    beta = self.sarsa_beta_scheduler.get_eps()
                    ilb, iub = self.relaxed_sarsa_model.compute_bounds(IBP=True, method=None)
                    if beta < 1:
                        clb, cub = self.relaxed_sarsa_model.compute_bounds(IBP=False, method='backward')
                        lb = beta * ilb + (1 - beta) * clb
                        ub = beta * iub + (1 - beta) * cub
                    else:
                        lb = ilb
                        ub = iub
                    # Output dimension is 1. Remove the extra dimension and keep only the batch dimension.
                    lb = lb.squeeze(-1)
                    ub = ub.squeeze(-1)
                    diff = torch.max(ub - q, q - lb)
                    reg_loss = self.params.SARSA_REG * (diff * diff).mean()
                    sarsa_loss = q_loss + reg_loss
                    reg_loss = reg_loss.item()
                else:
                    reg_loss = 0.0
                    sarsa_loss = q_loss
                sarsa_loss.backward()
                self.sarsa_opt.step()
            print(f'q_loss={q_loss.item():.6g}, reg_loss={reg_loss:.6g}, sarsa_loss={sarsa_loss.item():.6g}')

        if self.ANNEAL_LR:
            self.sarsa_scheduler.step()
        # print('value:', self.val_model(saps.states).mean().item())

        return q_loss, q.mean()

    def sarsa_step(self):
        '''
        Take a training step, by first collecting rollouts, and 
        taking a value function step.

        Inputs: None
        Returns: 
        - The current reward from the policy (per actor)
        '''
        print("-" * 80)
        start_time = time.time()

        num_saps = self.num_steps
        state_action_pairs, avg_ep_reward, avg_ep_length = self.collect_state_action_pairs(num_saps, test=True)
         
        sarsa_loss, q = self.sarsa_steps(state_action_pairs)
        print("Sarsa Loss:", sarsa_loss.item())
        print("Q:", q.item())
        print("Time elapsed (s):", time.time() - start_time)
        sys.stdout.flush()
        sys.stderr.flush()

        self.n_steps += 1
        return avg_ep_reward

    def run_test(self, num_steps=2000):
        
        with torch.no_grad():
            ep_length, ep_reward = self.run_test_trajectories(num_steps=num_steps)
            print(f"Episode reward: {ep_reward} | episode length: {ep_length}")
            
        return ep_length, ep_reward

    def run_test_trajectories(self, num_steps):

        # Reset env and get first state ----------------------------------------
        env = self.env
        
        # Initialize -----------------------------------------------------------
        rewards = torch.zeros((1, num_steps))

        actions_shape = (1, num_steps) + (self.NUM_ACTIONS,)
        actions = torch.zeros(actions_shape)
        action_means = torch.zeros(actions_shape)

        initial_state = cu_tensorize(env.reset())
        states_shape = (1, num_steps+1) + (initial_state.shape[0],)
        states =  torch.zeros(states_shape)

        states[:, 0, :] = initial_state
        last_states = initial_state.unsqueeze(0)
        
        # Take steps and collect
        completed_episode_info = []
        for t in range(num_steps):

            # Apply attack
            maybe_attacked_last_states = self.apply_attack(last_states)
            
            # Forward policy to get action distribution
            action_pds = self.policy_model(maybe_attacked_last_states)

            # Get means and stds
            next_action_means, next_action_stds = action_pds

            # Check if the attack is within eps range
            if self.params.ATTACK_METHOD != "none":
                
                # Max deviation from last states
                max_eps = (maybe_attacked_last_states - last_states).abs().max()

                # Attack epsilon range
                attack_eps = float(self.params.ATTACK_EPS)
                # TODO: removed as logic has changed
                # if max_eps > attack_eps + 1e-5:
                #     raise RuntimeError(f"{max_eps} > {attack_eps}. Attack implementation problem.")
            
            # Sample action according to distribution
            next_actions = self.policy_model.sample(action_pds)
            
            # Take step
            (done_info, next_rewards,
             next_states, next_not_dones) = self.step_wrapper(next_actions,
                                                              env)

            # Update history ---------------------------------------------------
            pairs = [
                (rewards, next_rewards),
                (actions, next_actions),
                (action_means, next_action_means),
                (states, next_states),
            ]
            
            # Filling the totals (defined above) for the trajectories
            last_states = next_states.unsqueeze(0)
            for total_array, new_value in pairs:
                if total_array is states:
                    # Next states, stores in the next position
                    total_array[:, t+1] = new_value
                else:
                    # The current action taken, and reward received
                    total_array[:, t] = new_value
            
            # If some of the actors finished AND this is not the last step
            # OR some of the actors finished AND we have no episode information
            if done_info != None:
                completed_episode_info.extend(done_info)
                break
        
        # Return length and reward
        if len(completed_episode_info) > 0:
            ep_length, ep_reward = completed_episode_info
        else:
            ep_length = np.nan
            ep_reward = np.nan       
        
        return ep_length, ep_reward