import os
import re
import sys
import time
import random
import torch
import cpprb
import numpy as np
import gymnasium as gym
from datetime import datetime
from src_dqn.attacks import attack
from src_dqn.models import model_setup
from src_dqn.config_loading import load_config, argparser
from src_dqn.epsilon_scheduler import EpsilonScheduler
from src_dqn.utils import CudaTensorManager, Logger, update_target, plot_array, save_arrays
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from torch.utils.tensorboard import SummaryWriter

USE_CUDA = torch.cuda.is_available()

# The robust part
def get_logits_lower_bound(model, state, state_ub, state_lb, eps, C, beta):
    
    # Define perturbation. Linf perturbation to input data. Just puts everything together
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=state_lb, x_U=state_ub)
    # Make the input a BoundedTensor with the pre-defined perturbation. state and bounds together
    bnd_state = BoundedTensor(state, ptb)
    pred = model(bnd_state, method_opt="forward") # Forward
    # Compute LiRPA bounds without the backward mode propagation.
    # Returns lower and upper bound.
    logits_ilb, _ = model.features.compute_bounds(C=C, IBP=True, method=None) # TODO: check why not needs ".features"
    if beta < 1e-5: # For small beta, towards the end, take logits as they are
        logits_lb = logits_ilb
    else: # For bigger beta, at the beginning, use a weighted average of 2 methods.
        # Compute LiRPA bounds using the backward mode bound propagation.
        # Returns lower and upper bound.
        logits_clb, _ = model.features.compute_bounds(IBP=False, C=C, method="backward", bound_upper=False)
        logits_lb = beta * logits_clb + (1-beta) * logits_ilb
    return logits_lb


class TimeLogger(object):
    def __init__(self):
        self.time_logs = {}

    def log_time(self, time_id, time):
        if time_id not in self.time_logs:
            self.time_logs[time_id] = 0.0
        self.time_logs[time_id] += time

    def __call__(self, time_id, time):
        self.log_time(time_id, time)

    def clear(self):
        self.time_logs = {}

    def print(self):
        print_str = ""
        for t in self.time_logs:
            print_str += "{}={:.4f} ".format(t, self.time_logs[t])
        print(print_str + "\n")

log_time = TimeLogger()


def logits_margin(logits, y):
    comp_logits = logits - torch.zeros_like(logits).scatter(1, torch.unsqueeze(y, 1), 1e10)
    sec_logits, _ = torch.max(comp_logits, dim=1)
    margin = sec_logits - torch.gather(logits, 1, torch.unsqueeze(y, 1)).squeeze(1)
    margin = margin.sum()
    return margin


def compute_td_loss(current_model, target_model, batch_size, replay_buffer, optimizer, gamma, memory_mgr, robust, **kwargs):
    t = time.time()

    result = replay_buffer.sample(batch_size)
    state, action, reward, next_state, done = result['obs'], result['act'], result['rew'], result['next_obs'], result['done']
 
    action = action.transpose()[0].astype(int)
    reward = reward.transpose()[0].astype(int)
    done = done.transpose()[0].astype(int)
    log_time('sample_time', time.time() - t)

    t = time.time()


    state, next_state, action, reward, done = memory_mgr.get_cuda_tensors(state, next_state, np.array(action), np.array(reward), np.array(done))
    
    
    optimizer.zero_grad()

    state = state.to(torch.float)
    next_state = next_state.to(torch.float)
    beta = kwargs.get('beta', 0)

    if robust:
        cur_q_logits = current_model(state, method_opt="forward")
        tgt_next_q_logits = target_model(next_state, method_opt="forward")
    else:
        cur_q_logits = current_model(state)
        tgt_next_q_logits = target_model(next_state)
    if robust:
        eps_adversary = kwargs['eps_adversary']
    cur_q_value = cur_q_logits.gather(1, action.unsqueeze(1)).squeeze(1)

    tgt_next_q_value = tgt_next_q_logits.max(1)[0]
    expected_q_value = reward + gamma * tgt_next_q_value * (1 - done)

    # Compute loss
    loss_fn  = torch.nn.MSELoss()
    loss = loss_fn(cur_q_value, expected_q_value.detach())

    # Mean Q value
    batch_cur_q_value = torch.mean(cur_q_value)
    batch_exp_q_value = torch.mean(expected_q_value)
    loss = loss.mean()
    td_loss = loss.clone()

    if robust:
        if eps_adversary < np.finfo(np.float32).tiny:
            reg_loss = torch.zeros(state.size(0))
            if USE_CUDA:
                reg_loss = reg_loss.cuda()
        else:

            sa = kwargs.get('sa', None)
            pred = cur_q_logits # predicted prob of actions
            labels = torch.argmax(pred, dim=1).clone().detach() # action with highest prob
            c = torch.eye(current_model.num_actions).type_as(state)[labels].unsqueeze(1) - torch.eye(current_model.num_actions).type_as(state).unsqueeze(0) # one hot the actions, make 4 rows (out of 1) and subtract the diag matrix
            I = (~(labels.data.unsqueeze(1) == torch.arange(current_model.num_actions).type_as(labels.data).unsqueeze(0))) # batch x action with boolean with False at the location where the action is the label one
            c = (c[I].view(state.size(0), current_model.num_actions-1, current_model.num_actions)) # keeps c rows which are not 0 (-1 at diag and 1 at real action)
            sa_labels = sa[labels.cpu().numpy()] # TODO: check cuda - if action was 2 it would be 0,1,3
            lb_s = torch.zeros(state.size(0), current_model.num_actions)
            if USE_CUDA:
                labels = labels.cuda()
                c = c.cuda()
                sa_labels = sa_labels.cuda()
                lb_s = lb_s.cuda()


            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_min = torch.Tensor(kwargs['state_min']).to(device)
            state_max = torch.Tensor(kwargs['state_max']).to(device)
            range_percent = eps_adversary * (state_max-state_min)
            sphere_min = range_percent * torch.zeros(state.data.size()).to(device) - 0.5 * range_percent
            sphere_max = range_percent * torch.ones(state.data.size()).to(device) - 0.5 * range_percent
            state_ub = torch.clamp(state + sphere_max, max=state_max)
            state_lb = torch.clamp(state + sphere_min, min=state_min)

            lb = get_logits_lower_bound(current_model, state, state_ub, state_lb, range_percent, c, beta)
            lb = lb_s.scatter(1, sa_labels, lb) # Put lb at not optimal action locations
            reg_loss = torch.nn.CrossEntropyLoss()(-lb, labels)

        reg_loss = reg_loss.mean()
        kappa = kwargs['kappa']
        loss += kappa * reg_loss

    # Back propagation
    loss.backward()
    # Update weights
    optimizer.step()

    log_time('nn_time', time.time() - t)

    # Return tuple if not robust
    result = (loss, td_loss, batch_cur_q_value, batch_exp_q_value)
    
    # If robust more elements are returned
    if robust:
        result += (reg_loss,)
    
    return result


def mini_test(model, env_id, logger, seeds, attack_flag=False, attack_config={}):
    
    logger.log('MINI TEST! ' + '#'*20)
    
    # Make env
    env = gym.make(env_id)
    
    state, _ = env.reset(seed=seeds[0])

    all_rewards = []
    episode_idx = 0
    episode_reward = 0
    episode_step = 0
    while episode_idx < len(seeds)-1:
        
        # Take action and step
        state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
        if attack_flag:
            state_tensor = attack(model, state_tensor, attack_config)
        action = model.act(state_tensor)[0]
        next_state, reward, termination, truncation, _ = env.step(action)
        done = termination
        state = next_state
        episode_reward += reward
        episode_step += 1

        if done or episode_step >= 1000:
            
            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_step = 0
            episode_idx += 1
            state, _ = env.reset(seed=seeds[episode_idx])

    return all_rewards


def main(config, version):
    
    # Extract main sections from the config
    env_id = config['env_id']
    train_config = config['training_config']

    # Create experiment folder
    date_time = datetime.now().strftime("%Y%d%m_%H%M")
    output_folder = os.path.join('output_dqn', f"{env_id}_{version}", date_time)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create log file with the output printed over the iterations
    train_log = os.path.join(output_folder, 'train.log')
    logger = Logger(open(train_log, "w"))
    logger.log('Command line:', " ".join(sys.argv[:]))
    logger.log(config)

    # Initialize writer for tensorflow and write the hyperparameters
    writer = SummaryWriter(output_folder)
    list_args_key_value = [f"|{key}|{str(value)}|" 
                                for key, value in train_config.items()]
    args_key_value = '\n'.join(list_args_key_value)
    writer.add_text(
                tag="Hyperparameters",
                text_string=f"|Parameter|Value|\n|-|-|\n{args_key_value}"
                )

    # Make environment
    env = gym.make(env_id)

    # Set seeds
    seed = train_config['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Reset environment
    state, _ = env.reset(seed=seed)
    # Log environment related info
    logger.log(f"observations shape: {env.observation_space.shape}")
    logger.log(f"num of actions: {env.action_space.n}")

    # Check robust related info
    robust = train_config['robust']
    print('robust', robust)
    adv_train = train_config['adv_train']
    bound_solver = train_config['bound_solver']
    attack_config = {}

    # Adversarial training
    if adv_train:
        attack_config = train_config["attack_config"]
        adv_ratio = train_config['adv_ratio']
        if adv_train:
            logger.log('adversarial examples for training, adv ratio:', adv_ratio)
    
    # Robust or adverarial training -> Adversary Epsilon Scheduler
    if robust or adv_train:

        eps_scheduler_adversary = EpsilonScheduler(
                            epsilon_start=train_config['eps_start_value_adv'],
                            epsilon_final=train_config['eps_end_value_adv'],
                            duration=train_config['eps_schedule_length_adv'],
                            start_idx=train_config['eps_schedule_start_adv'])

    # Action Epsilon Scheduler
    act_epsilon_scheduler = EpsilonScheduler(
                            epsilon_start=train_config['act_epsilon_start'],
                            epsilon_final=train_config['act_epsilon_final'],
                            duration=train_config['act_epsilon_duration'],
                            start_idx=train_config['buffer_params']['replay_initial'])
    
    # Model Setup - including the robust part
    current_model = model_setup(env=env,
                                robust_model=robust,
                                use_cuda=USE_CUDA)
    target_model = model_setup(env=env,
                               robust_model=robust,
                               use_cuda=USE_CUDA)

    # Setup optimizer
    optimizer = torch.optim.Adam(current_model.parameters(),
                                 lr=train_config['lr'])
    
    # Do not evaluate gradient for target model
    for param in target_model.features.parameters():
        param.requires_grad = False

    # Experience Replay
    replay_buffer = cpprb.ReplayBuffer(train_config['buffer_params']['buffer_capacity'],
                    {"obs": {"shape": state.shape, "dtype": state.dtype},
                     "act": {"shape": 1, "dtype": np.uint8},
                     "rew": {},
                     "next_obs": {"shape": state.shape, "dtype": state.dtype},
                     "done": {}})
    
    # Set the target model to have the same weights as the current model
    update_target(current_model=current_model,
                  target_model=target_model,
                  tau=0)
    
    # Optimized cuda memory management
    memory_mgr = CudaTensorManager(state_shape=state.shape,
                                   batch_size=train_config['batch_size'],
                                   use_cuda=USE_CUDA,
                                   dtype=state.dtype)

    # Related to robust version
    sa = None
    kappa = None
    if robust:
        kappa = train_config['kappa']
        reg_losses = []
        sa = np.zeros(shape=(current_model.num_actions, current_model.num_actions - 1),
                      dtype = np.int32)
        for i in range(sa.shape[0]): # Each row is missing the index of one of the actions
            for j in range(sa.shape[1]):
                if j < i:
                    sa[i][j] = j
                else:
                    sa[i][j] = j + 1
        sa = torch.LongTensor(sa)

    # Initializations
    losses = []
    td_losses = []
    batch_cur_q = []
    batch_exp_q = []
    all_rewards = []
    act_epsilon_list = []
    eps_adversary_list = []
    betas = []
    

    best_test_reward = -float('inf')
    buffer_stored_size = 0
    if adv_train:
        attack_count = 0
        success_count = 0

    start_time = time.time()
    period_start_time = time.time()

    episode_count = 0
    episode_reward = 0
    episode_dur = 0
    
    beta = np.nan

    last_printed = 0
    last_run_at = 0
    last_saved = 0

    # Action epsilon scheduling
    act_epsilon = np.nan
    act_epsilon = act_epsilon_scheduler.get(0)
    
    # Epsilon attack - controls entity attack not percentage
    eps_adversary = 0
    if adv_train or robust:
        eps_adversary = eps_scheduler_adversary.get(0)
    
    # Main Loop of Iterations
    iter_idx = -1
    while episode_count <= train_config["max_episodes"]:
    # for iter_idx in range(start_idx, train_config['num_iters'] + 1):
        
        iter_idx += 1
        # Time Logging
        iteration_start = time.time()
        t = time.time()

        # ACTION ---------------------------------------------------------------
        # If adversarial training, epsilon not nan and greater than "tiny"
        if adv_train and eps_adversary != np.nan and eps_adversary >= np.finfo(np.float32).tiny:
            # Save original state
            original_state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
            attack_config['params']['epsilon'] = eps_adversary
            # If random below adversarial training ratio -> Attack!
            if random.random() < adv_ratio:
                attack_count += 1
                # Attacked state
                state_tensor = attack(current_model, original_state_tensor, attack_config)
                # If action of attacked state different than the one 
                # that would be take with original state increase success count
                if current_model.act(state_tensor)[0] != current_model.act(original_state_tensor)[0]:
                    success_count += 1

            # Otherwise the state remains unperturbed
            else:
                state_tensor = original_state_tensor
            # The action is taken
            action = current_model.act(state_tensor, act_epsilon)[0]
        # If there is no adversarial training or epsilon is nan/tiny
        else:
            with torch.no_grad():
                # The state is the actual state
                state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
                # Keep a copy of the original state
                original_state_tensor = torch.clone(state_tensor)
                # Take an action
                action = current_model.act(state_tensor, act_epsilon)[0]

        # Log the time for the action selection
        log_time('act_time', time.time() - t)

        # STEP -----------------------------------------------------------------
        next_state, reward, termination, truncation, _ = env.step(action)
        done = termination


        # BUFFER ---------------------------------------------------------------
        t = time.time()
        replay_buffer.add(obs=state,
                          act=action,
                          rew=reward,
                          next_obs=next_state,
                          done=done)
        
        log_time('save_time', time.time() - t)

        # Get the current size / length of the replay buffer
        buffer_stored_size = replay_buffer.get_stored_size()

        t = time.time()

        # LEARN ----------------------------------------------------------------
        # If the current buffer size is the same as the initial replay buffer
        # And at certain frequency
        # Compute the TD loss
        if (buffer_stored_size > train_config['buffer_params']['replay_initial']
            and buffer_stored_size % 5 == 0):
            # If robust set beta
            if robust:
                convex_final_beta = train_config['convex_final_beta']
                convex_start_beta = train_config['convex_start_beta']
                max_eps = train_config['eps_end_value_adv']
                beta = (max_eps - eps_adversary * (1.0 - convex_final_beta)) / max_eps * convex_start_beta # Beta goes from 1 to 0 depending on eps_adversary
            else:
                beta = np.nan
            # Compute the TD loss
            result = compute_td_loss(current_model,
                                  target_model,
                                  train_config['batch_size'],
                                  replay_buffer,
                                  optimizer,
                                  train_config['gamma'],
                                  memory_mgr,
                                  robust,
                                  eps_adversary=eps_adversary,
                                  beta=beta,
                                  sa=sa,
                                  kappa=kappa,
                                  dtype=state.dtype,
                                  env_id=env_id,
                                  bound_solver=bound_solver,
                                  attack_config=attack_config,
                                  state_min=train_config['attack_config']['params']['state_min'],
                                  state_max=train_config['attack_config']['params']['state_max'])
            
            # Extract information
            loss, td_loss, batch_cur_q_value, batch_exp_q_value = result[0], result[1], result[2], result[3]
            # If robust we get also the regularization loss
            if robust:
                reg_loss = result[-1]
                reg_losses.append(reg_loss.data.item())


            # Update target network --------------------------------------------
            update_target(current_model, target_model, tau=1e-3)

            # Append losses and q values
            losses.append(loss.data.item())
            td_losses.append(td_loss.data.item())
            batch_cur_q.append(batch_cur_q_value.data.item())
            batch_exp_q.append(batch_exp_q_value.data.item())

            writer.add_scalar("Plots/loss", loss, iter_idx)
            writer.add_scalar("Plots/td_loss", td_loss, iter_idx)
            writer.add_scalar("Plots/batch_cur_q_value", batch_cur_q_value, iter_idx)
            writer.add_scalar("Plots/batch_exp_q_value", batch_exp_q_value, iter_idx)

        log_time('loss_time', time.time() - t)

        # Update states and reward ---------------------------------------------
        t = time.time()
        state = next_state
        episode_reward += reward

        episode_dur +=1

        # Update Epsilons
        # Action epsilon scheduling 
        # TODO: check
        act_epsilon = act_epsilon_scheduler.get(iter_idx)
        if adv_train or robust:
            eps_adversary = eps_scheduler_adversary.get(iter_idx)
        
        act_epsilon_list.append(act_epsilon)
        eps_adversary_list.append(eps_adversary)
        betas.append(beta)

        writer.add_scalar("Plots/act_epsilon", act_epsilon, iter_idx)
        writer.add_scalar("Plots/eps_adversary", eps_adversary, iter_idx)
        writer.add_scalar("Plots/beta", beta, iter_idx)


        # DONE -----------------------------------------------------------------
        # If done reset the env and the episode reward
        if done or episode_dur >= 1000:
            state, _ = env.reset()
            all_rewards.append(episode_reward)
            writer.add_scalar("Plots/episode_reward", episode_reward, episode_count)
            episode_reward = 0
            episode_count += 1
            episode_dur = 0

        log_time('env_time', time.time() - t)

        # All kinds of result logging ------------------------------------------

        # If frequency print frame is met or 
        # it is the end frame or
        # robust and right before and after the epsilon schedule start or
        # the buffer size is close to the the replay initial value
        
        if (episode_count != last_printed and
            (episode_count % train_config['print_epi'] == 0 or \
             episode_count==train_config['max_episodes'])):
            last_printed = episode_count
            
            # IDX
            logger.log(f"\nIteration {iter_idx}.")
            # EPISODE COUNT
            logger.log(f"Completed {episode_count} episodes.")
            # EPS
            logger.log(f"- EPS ACTION: {act_epsilon:.4f}")
            # TIME
            total_time = time.time() - start_time
            epoch_time = time.time() - period_start_time
            logger.log(f"- TIME - total: {total_time:.4f}, epoch: {epoch_time:.4f}")
            # LOSS
            last_loss = losses[-1] if losses else np.nan
            avg_loss = np.average(losses[:-51:-1]) if losses else np.nan
            logger.log(f"- LOSS - last: {last_loss:.4f}, avg: {avg_loss:.4f}")
            # REWARD
            last_reward = all_rewards[-1] if all_rewards else np.nan
            avg_reward = np.average(all_rewards[:-51:-1]) if all_rewards else np.nan
            logger.log(f"- REWARD - last episode: {last_reward:.4f}, avg 50 epi: {avg_reward:.4f}")
            # TD LOSS AND Q
            logger.log('- last td loss: {:.6g}, avg td loss: {:.6g}'.format(
                    td_losses[-1] if td_losses else np.nan,
                    np.average(td_losses[:-51:-1]) if td_losses else np.nan))
            logger.log('- last batch cur q: {:.6g}, avg batch cur q: {:.6g}'.format(
                    batch_cur_q[-1] if batch_cur_q else np.nan,
                    np.average(batch_cur_q[:-51:-1]) if batch_cur_q else np.nan))
            logger.log('- last batch exp q: {:.6g}, avg batch exp q: {:.6g}'.format(
                    batch_exp_q[-1] if batch_exp_q else np.nan,
                    np.average(batch_exp_q[:-51:-1]) if batch_exp_q else np.nan))
            
            # Just for robust --------------------------------------------------
            if robust:
                logger.log('EPS ATTACK: {:.6g}'.format(eps_adversary))
                logger.log('BETA: {:.6g}'.format(beta))
                logger.log('last cert reg loss: {:.6g}, avg cert reg loss: {:.6g}'.format(
                    reg_losses[-1] if reg_losses else np.nan,
                    np.average(reg_losses[:-51:-1]) if reg_losses else np.nan))
                logger.log('KAPPA: {:.6g}'.format(kappa))
                abs_diff = abs(state_tensor.cpu().numpy()-original_state_tensor.cpu().numpy())
                logger.log(f'ATTACK: l1 norm: {np.sum(abs_diff)}, l2 norm: {np.linalg.norm(abs_diff)}, linf norm: {np.max(abs_diff)}')
            
            # For adversarial training
            if adv_train:
                logger.log('EPS ATTACK: {:.6g}'.format(eps_adversary))
                success_rate = success_count*1.0/attack_count if attack_count>0 else np.nan
                logger.log(f'This batch attacked: {attack_count}, success: {success_count}, attack success rate: {success_rate:.6g}')
                attack_count = 0
                success_count = 0
                abs_diff = abs(state_tensor.cpu().numpy()-original_state_tensor.cpu().numpy())
                logger.log(f'ATTACK: l1 norm: {np.sum(abs_diff)}, l2 norm: {np.linalg.norm(abs_diff)}, linf norm: {np.max(abs_diff)}')

            period_start_time = time.time()
            log_time.print()
            log_time.clear()

        # Save plots/arrays (reward/loss) and save model
        if (episode_count != last_saved and 
            (episode_count % train_config['save_epi_interval'] == 0 or 
             episode_count == train_config['max_episodes'])):
            
            last_saved = episode_count
            # Plot 
            plot_array(iter_idx, episode_count, all_rewards, 'rewards', output_folder)
            plot_array(iter_idx, episode_count, losses, 'losses', output_folder)
            plot_array(iter_idx, episode_count, batch_cur_q, 'batch_cur_q', output_folder)
            plot_array(iter_idx, episode_count, batch_exp_q, 'batch_exp_q', output_folder)
            plot_array(iter_idx, episode_count, eps_adversary_list, 'eps_adversary', output_folder)
            plot_array(iter_idx, episode_count, act_epsilon_list, 'eps_action', output_folder)
            plot_array(iter_idx, episode_count, betas, 'beta', output_folder)

            # Save arrays
            arrays_dict = {'losses': losses,
                           'batch_cur_q': batch_cur_q,
                           'batch_exp_q': batch_exp_q}
            save_arrays(arrays_dict, 'losses_and_q', output_folder)
            arrays_dict = {'eps_adversary_list': eps_adversary_list,
                           'act_epsilon_list': act_epsilon_list,
                           'betas': betas}
            save_arrays(arrays_dict, 'eps_and_beta', output_folder)
            arrays_dict = {'rewards': all_rewards}
            save_arrays(arrays_dict, 'rewards', output_folder)
                        
            # Save model
            if episode_count > train_config['save_epi_start']:
                path_models = os.path.join(output_folder, 'models')
                os.makedirs(path_models, exist_ok=True)
                torch.save(current_model.features.state_dict(),
                        f'{path_models}/idx_{iter_idx}_epi_{episode_count}.pth')

        
        # Minitest
        if (episode_count % train_config['save_epi_interval'] == 0 and
            episode_count > train_config['save_epi_start']
            and episode_count != last_run_at):
            
            seeds = [random.randint(0, sys.maxsize) for i in range(20)]

            # Attack
            test_rewards = mini_test(model=current_model,
                                     env_id=env_id,
                                     logger=logger,
                                     seeds=seeds,
                                     attack_flag=True,
                                     attack_config=train_config["attack_config"])
            logger.log(f'All rewards: {np.array(test_rewards).round(decimals=2)}')
            mean_reward = np.mean(test_rewards)
            median_reward = np.median(test_rewards)
            std_reward = np.std(test_rewards)
            logger.log(f'Mean reward: {mean_reward:6g}')
            logger.log(f'Median reward: {median_reward:6g}')
            logger.log(f'Std reward: {std_reward:6g}')
            last_run_at = episode_count

            # No attack
            test_rewards = mini_test(model=current_model,
                                     env_id=env_id,
                                     logger=logger,
                                     seeds=seeds,
                                     attack_flag=False)
            logger.log(f'All rewards: {np.array(test_rewards).round(decimals=2)}')
            mean_reward = np.mean(test_rewards)
            median_reward = np.median(test_rewards)
            std_reward = np.std(test_rewards)
            logger.log(f'Mean reward: {mean_reward:6g}')
            logger.log(f'Median reward: {median_reward:6g}')
            logger.log(f'Std reward: {std_reward:6g}')

            if median_reward >= best_test_reward:
                best_test_reward = median_reward
                logger.log(f'NEW best reward {best_test_reward:6g} achieved!')
                path_best_models = os.path.join(output_folder, 'best_models')
                os.makedirs(path_best_models, exist_ok=True)
                torch.save(current_model.features.state_dict(),
                           f'{path_best_models}/idx_{iter_idx}_epi_{episode_count}.pth')

        log_time.log_time('total', time.time() - iteration_start)

    writer.close()


if __name__ == "__main__":

    # Load args - especially config file
    args = argparser()
    config = load_config(args, default_config='config_dqn/defaults.json')
    version = os.path.basename(args.config).split('_')[1].split('.')[0]
    main(config, version)
