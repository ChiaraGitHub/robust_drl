# General imports
import os
import sys
import random
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# From other scripts
from src_ppo.config_loading import load_config, argparser
from src_ppo.utils import Logger
from src_ppo.agent import Trainer


def main(config, version):

    # Create experiment folder -------------------------------------------------
    
    # Extract environment id from config
    env_id = config['env_id']

    # Name and create folder (output/type algo/datetime)
    date_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_folder = os.path.join('output_ppo', f"{env_id}_{version}", date_time)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Output folder: {output_folder}")

    # Create folders for plots, checkpoints, best model
    path_plots = os.path.join(output_folder, 'plots')
    os.makedirs(path_plots, exist_ok=True)
    path_checkpoints = os.path.join(output_folder, 'checkpoints')
    os.makedirs(path_checkpoints, exist_ok=True)
    path_best_models = os.path.join(output_folder, 'best_model')
    os.makedirs(path_best_models, exist_ok=True)

    # Create train.log file for logging
    train_log = os.path.join(output_folder, 'train.log')
    logger = Logger(open(train_log, "w"))
    logger.log('Command line:', " ".join(sys.argv[:]))
    logger.log(config)

    # Tensorboard logging
    writer = SummaryWriter(output_folder)
    list_args_key_value = [f"|{key}|{str(value)}|" 
                                for key, value in config.items()]
    args_key_value = '\n'.join(list_args_key_value)
    writer.add_text(
                tag="Hyperparameters",
                text_string=f"|Parameter|Value|\n|-|-|\n{args_key_value}"
                )

    # Set seeds ----------------------------------------------------------------
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Training iterations ------------------------------------------------------
    trainer = Trainer(config, logger, writer)
    rewards = []
    lengths = []
    best_reward = -np.inf
    for i in range(config['train_iterations']):
        
        logger.log(f'Step {i}')

        # Get new reward
        mean_reward, mean_length = trainer.train_step_entrypoint()
        rewards.append(mean_reward)
        lengths.append(mean_length)
    
        # Update REWARD plots
        plt.figure(figsize=(10,5))
        plt.title(f'Reward')
        plt.scatter(range(len(rewards)), rewards)
        plt.grid()
        path_plots = os.path.join(output_folder, 'plots')
        os.makedirs(path_plots, exist_ok=True)
        plt.savefig(f'{path_plots}/rewards.png')
        plt.close()

        # Update LENGTH plots
        plt.figure(figsize=(10,5))
        plt.title(f'Episode Length')
        plt.scatter(range(len(lengths)), lengths)
        plt.grid()
        path_plots = os.path.join(output_folder, 'plots')
        os.makedirs(path_plots, exist_ok=True)
        plt.savefig(f'{path_plots}/lengths.png')
        plt.close()

        # Save checkpoints
        if config['save_iters'] > 0 and i % config['save_iters'] == 0 and i != 0:
            
            # Compute and print average last rewards
            mean_last_rewards = np.nanmean(np.array(rewards)[-50:])
            logger.log(f'Mean last rewards: {mean_last_rewards:.5g}')

            # Save checkpoints
            path_this_ckpt = f'{path_checkpoints}/iter_{i}/'
            os.makedirs(path_this_ckpt)
            torch.save(trainer.val_model.state_dict(), f'{path_this_ckpt}/val_model')
            torch.save(trainer.policy_model.state_dict(), f'{path_this_ckpt}/policy_model')
            torch.save(trainer.POLICY_ADAM.state_dict(), f'{path_this_ckpt}/policy_opt')
            torch.save(trainer.val_opt.state_dict(), f'{path_this_ckpt}/val_opt')
            torch.save(trainer.env, f'{path_this_ckpt}/envs')

        # Save best models
        if mean_reward > best_reward and i > config['save_iters']:
            best_reward = mean_reward
            path_this_best_model = f'{path_best_models}/iter_{i}'
            os.makedirs(path_this_best_model)
            torch.save(trainer.val_model.state_dict(), f'{path_this_best_model}/val_model')
            torch.save(trainer.policy_model.state_dict(), f'{path_this_best_model}/policy_model')
            torch.save(trainer.POLICY_ADAM.state_dict(), f'{path_this_best_model}/policy_opt')
            torch.save(trainer.val_opt.state_dict(), f'{path_this_best_model}/val_opt')
            torch.save(trainer.env, f'{path_this_best_model}/envs')

    return


if __name__ == '__main__':

    # Load args - especially config file
    args = argparser()
    config = load_config(args, default_config='')

    # Versions are nat/adv/cov
    version = os.path.basename(args.config).split('_')[1]

    main(config, version)