# General imports
import random
import gymnasium as gym
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box as Continuous


class Env:

    def __init__(self,
                 game):
        
        self.env = gym.make(game, continuous=True)

        # Environment type
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Continuous

        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1 # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]
        
        # Number of features
        assert len(self.env.observation_space.shape) == 1
        s, _ = self.env.reset()
        self.num_features = s.shape[0]

        # Running total reward (set to 0.0 at resets)
        self.total_reward = 0.0
   
    def reset(self):

        # Reset the state, reward and counter
        start_state, _ = self.env.reset(seed=random.getrandbits(31))
        self.total_reward = 0.0
        self.counter = 0.0

        return start_state

    def step(self, action):
        
        # Step and update total reward and counter
        state, reward, terminated, truncated, info = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward
        self.counter += 1

        # In info provide length and total reward
        if is_done:
            info['done'] = (self.counter, self.total_reward)
        
        return state, reward, is_done, info

    