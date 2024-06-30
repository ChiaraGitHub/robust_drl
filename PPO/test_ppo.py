import os
import sys
import random
import json
import torch
import numpy as np
import pandas as pd

from src_ppo.config_loading import load_config, argparser
from src_ppo.agent import Trainer
from src_ppo.utils import Logger
from auto_LiRPA.eps_scheduler import LinearScheduler


def main(config):

    # Set seeds ----------------------------------------------------------------
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Make deterministic -------------------------------------------------------
    config["deterministic"] = True

    # Check if sarsa_enable, attack is none ------------------------------------
    if config['sarsa_enable']:
        assert config['attack_method'] == "none", \
            f"--train-sarsa only when --attack-method=none, got {config['attack_method']}"

    # Name and create folder
    output_folder_test = os.path.join(config['base_test_path'],
                                      "test_attack_" + config["attack_method"])
    os.makedirs(output_folder_test, exist_ok=True)
    

    # Create test.log file for logging
    test_log = os.path.join(output_folder_test, 'test.log')
    logger = Logger(open(test_log, "w"))
    logger.log('Command line:', " ".join(sys.argv[:]))
    logger.log(f"Output folder: {output_folder_test}")

    # Load/create agent from config file ---------------------------------------
    trainer = Trainer(config, logger, writer=None)
    path_model = os.path.join(config['base_test_path'], config['load_model'])
    logger.log('Loading pretrained model', path_model)
    trainer.policy_model.load_state_dict(torch.load(f'{path_model}/policy_model'))
    trainer.val_model.load_state_dict(torch.load(f'{path_model}/val_model'))
    trainer.POLICY_ADAM.load_state_dict(torch.load(f'{path_model}/policy_opt'))
    trainer.val_opt.load_state_dict(torch.load(f'{path_model}/val_opt'))
    trainer.envs = torch.load(f'{path_model}/envs')

    # Check noise/policy model stdev -------------------------------------------
    trainer.policy_model.log_stdev.data[:] = -100
    print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
    print('Gaussian noise in policy (after adjustment):')
    print(torch.exp(trainer.policy_model.log_stdev))

    # Testing ------------------------------------------------------------------
    rewards = []
    if config['sarsa_enable']:
        num_steps = config['sarsa_steps']
        # learning rate scheduler: linearly annealing learning rate after 
        lr_decrease_point = num_steps * 2 / 3
        decreasing_steps = num_steps - lr_decrease_point
        lr_sch = lambda epoch: 1.0 if epoch < lr_decrease_point else (decreasing_steps - epoch + lr_decrease_point) / decreasing_steps
        # robust training scheduler. Currently using 1/3 epochs for warmup, 1/3 for schedule and 1/3 for final training.
        eps_start_point = int(num_steps * 1 / 3)
        robust_eps_scheduler = LinearScheduler(config['sarsa_eps'], f"start={eps_start_point},length={eps_start_point}")
        robust_beta_scheduler = LinearScheduler(1.0, f"start={eps_start_point},length={eps_start_point}")
        # reinitialize value model, and run value function learning steps.
        trainer.sarsa_setup(lr_schedule=lr_sch, eps_scheduler=robust_eps_scheduler, beta_scheduler=robust_beta_scheduler)
        # Run Sarsa training.
        for i in range(num_steps):
            print(f'Step {i+1} / {num_steps}, lr={trainer.sarsa_scheduler.get_last_lr()}')
            mean_reward = trainer.sarsa_step()
            rewards.append(mean_reward)
            # for w in p.val_model.parameters():
            #     print(f'{w.size()}, {torch.norm(w.view(-1), 2)}')
        # Save Sarsa model.
        saved_model = {
                'state_dict': trainer.sarsa_model.state_dict(),
                'metadata': config,
                }
        torch.save(saved_model,
                   os.path.join(config['base_test_path'],
                                config['sarsa_model_path']))
    
    else:
        
        # Run certain number of episodes ---------------------------------------
        all_rewards = []
        all_lengths = []
        for i in range(config['num_episodes']):
            
            logger.log('Episode %d / %d' % (i+1, config['num_episodes']))
            # Run test and collect reward and length
            ep_length, ep_reward = trainer.run_test()
            logger.log(f'Reward {ep_reward} - Length {ep_length}')
            all_rewards.append(ep_reward)
            all_lengths.append(ep_length)
            
        # Create attack folder name --------------------------------------------
        attack_dir = f'attack-eps-{config['attack_eps']}'
        if 'sarsa' in config['attack_method']:
            attack_dir += f'-sarsa_steps-{config['sarsa_steps']}'
            attack_dir += f'-sarsa_eps-{config['sarsa_eps']}'
            attack_dir += f'-sarsa_reg-{config['sarsa_reg']}'
            if 'action' in config['attack_method']:
                attack_dir += f'-attack_sarsa_action_ratio-{config['attack_sarsa_action_ratio']}'
        
        # Save rewards ---------------------------------------------------------
        save_path = os.path.join(output_folder_test, attack_dir)
        os.makedirs(save_path, exist_ok=True)
        df_rewards = pd.DataFrame([all_rewards, all_lengths],
                                  index=['rewards', 'lengths']).T
        df_rewards.to_csv(os.path.join(save_path,
                                       'rewards_lengths.csv'))
        
        # Save config ----------------------------------------------------------
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # Print results overview -----------------------------------------------
        logger.log('\n')
        logger.log('Rewards stats:')
        logger.log(f'mean: {np.mean(all_rewards)} +- std:{np.std(all_rewards)}')
        logger.log(f'min:{np.min(all_rewards)} / max:{np.max(all_rewards)}')

if __name__ == '__main__':

    # Load args - especially config file
    args = argparser()
    config = load_config(args, default_config='')
    
    # Run main
    main(config)
 
    