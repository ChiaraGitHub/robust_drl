import os
import gymnasium as gym
import sys
import random
import torch
import numpy as np
from src_dqn.config_loading import load_config, argparser
from src_dqn.attacks import attack
from src_dqn.models import model_setup
from src_dqn.utils import Logger, save_arrays

USE_CUDA = torch.cuda.is_available()

def main(config):

    # Extract main sections from the config
    env_id = config['env_id']
    test_config = config['test_config']
    attack_config = test_config["attack_config"]
    
    # Folder
    output_folder = test_config['load_folder_path']
    test_log = os.path.join(output_folder, 'test.log')
    logger = Logger(open(test_log, "w"))
    logger.log('Command line:', " ".join(sys.argv[:]))
    logger.log(config)
    
    # Make environment
    env = gym.make(env_id)

    # Reset the environment
    state, _ = env.reset()

    # Log environment related info
    logger.log(f"observations shape: {env.observation_space.shape}")
    logger.log(f"num of actions: {env.action_space.n}")

    # Model Setup
    model = model_setup(env=env,
                        robust_model=test_config['robust'],
                        use_cuda=USE_CUDA)

    # Loading model
    model_path = test_config['load_model_path']
    logger.log('model loaded from ' + model_path)
    model.features.load_state_dict(torch.load(model_path))
    num_episodes = test_config['num_episodes']
    max_steps_per_episode = test_config['max_steps_per_episode']

    # Resetting env
    seed = random.randint(0, sys.maxsize)
    logger.log('reseting env with seed', seed)
    state, _ = env.reset(seed=seed)

    episode_idx = 1
    episode_step = 1
    episode_reward = 0
    all_rewards = []

    for iter_idx in range(1, num_episodes * max_steps_per_episode + 1):

        state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)

        # Attack the state
        if test_config['attack']:
            state_tensor = attack(model, state_tensor, attack_config)

        # Take action
        action = model.act(state_tensor)[0]

        # Step
        next_state, reward, termination, truncation, _ = env.step(action)
        done = termination

        # Assign next state and update reward
        state = next_state
        episode_reward += reward

        if episode_step == max_steps_per_episode:
            logger.log('maximum number of frames reached in this episode, reset environment!')
            done = True

        if done:

            # Log at done
            logger.log(f'\nEpisode {episode_idx}/{num_episodes}')
            logger.log(f'Episode steps {episode_step}/{max_steps_per_episode}')
            logger.log('Last episode reward: {:.6g}, avg 10 episode reward: {:.6g}'.format(
                all_rewards[-1] if all_rewards else np.nan,
                np.average(all_rewards[:-11:-1]) if all_rewards else np.nan))
            logger.log(np.mean(all_rewards),'+-',np.std(all_rewards))

            # Resetting
            seed = random.randint(0, sys.maxsize)
            logger.log('reseting env with seed', seed)
            state, _ = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_step = 1
            episode_idx += 1
            if episode_idx > num_episodes:
                break
        else:
            episode_step += 1

    logger.log(np.mean(all_rewards),'+-',np.std(all_rewards))
    arrays_dict = {'rewards': all_rewards}
    if test_config['attack'] == True:
        version = 'attack'
    else:
        version = 'normal'
    save_arrays(arrays_dict, f'test_rewards_{version}', output_folder)

if __name__ == "__main__":
    
    # Load args - especially config file
    args = argparser()
    config = load_config(args, default_config='config_dqn/defaults.json')
    main(config)
