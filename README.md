# Vanilla/Adversarial/Robust DRL

## PPO
All code related to PPO can be found in the PPO folder.
The outputs are saved in the output_ppo subfolder. The sufixes "nat", "adv", "cov" denote the vanilla, adversarial and robust versions respectively.
Experiments subfolders are named based on the date and time when the experiment was run. 
In "best_models" the best models are stored. In "checkpoints" the models are stored at regular intervals. In "plots" some plots created during training are stored.
All subfolders starting with "test_" contain results of the testing phase.

#### Training

To re-train a vanilla version run from the PPO folder:
```python train_ppo.py --config config_ppo/Lunar_nat_ppo.json```

To re-train an adversarial version run from the PPO folder:
```python train_ppo.py --config config_ppo/Lunar_adv_ppo.json```

To re-train a robust version run from the PPO folder:
```python train_ppo.py --config config_ppo/Lunar_cov_ppo.json```

#### Testing

To test a vanilla version run from the PPO folder:
```python test_ppo.py --config config_ppo/Lunar_nat_ppo.json```

To test an adversarial version run from the PPO folder:
```python test_ppo.py --config config_ppo/Lunar_adv_ppo.json```

To test a robust version run from the PPO folder:
```python test_ppo.py --config config_ppo/Lunar_cov_ppo.json```


## DQN

All code related to DQN can be found in the DQN folder.
The outputs are saved in the output_dqn subfolder. The sufixes "nat", "adv", "cov" denote the vanilla, adversarial and robust versions respectively.
Experiments subfolders are named based on the date and time when the experiment was run.
In "best_models" the best models are stored. In "models" the models are stored at regular intervals. In "plots" some plots created during training are stored.
In the "csvs" subfolder some data collected during training and testing are saved in .csv format for further processing and inspection.

#### Training

To re-train a vanilla version run from the DQN folder:
```python train_dqn.py --config config_dqn/Lunar_nat_dqn.json```

To re-train an adversarial version run from the DQN folder:
```python train_dqn.py --config config_dqn/Lunar_adv_dqn.json```

To re-train a robust version run from the DQN folder:
```python train_dqn.py --config config_dqn/Lunar_cov_dqn.json```

#### Testing

To test a vanilla version run from the DQN folder:
```python test_dqn.py --config config_dqn/Lunar_nat_dqn.json```

To test an adversarial version run from the DQN folder:
```python test_dqn.py --config config_dqn/Lunar_adv_dqn.json```

To test a robust version run from the DQN folder:
```python test_dqn.py --config config_dqn/Lunar_cov_dqn.json```

# auto_LiRPA

The auto_LiRPA subfolders contain code taken from the following repository:
https://github.com/Verified-Intelligence/auto_LiRPA

