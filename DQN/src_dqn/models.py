import torch
import numpy as np

import sys
sys.path.append("././auto_LiRPA")
from auto_LiRPA import BoundedModule

class QNetwork(torch.nn.Module):
    
    def __init__(self, env, robust, use_cuda):
        super().__init__()
        self.env = env
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.robust = robust
        self.features = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_actions),
        )

        # Bounded Module in case of robust training
        if self.robust:
            dummy_input = torch.empty_like(torch.randn((1,) + self.input_shape))
            self.features = BoundedModule(self.features,
                                          dummy_input,
                                          device="cuda" if use_cuda else "cpu")

    def forward(self, *args, **kwargs):
        return self.features(*args, **kwargs)
    
    def act(self, state, epsilon=0):

        if self.robust:
            q_value = self.forward(state, method_opt='forward')
        else:
            q_value = self.forward(state)

        # epsilon greedy action selection
        action  = q_value.max(1)[1].data.cpu().numpy()
        mask = np.random.choice(np.arange(0, 2), p=[1-epsilon, epsilon])
        action = (1-mask) * action + \
                 mask * np.random.randint(self.env.action_space.n, size=state.size()[0])
        
        return action
   
def model_setup(
                env,
                robust_model: bool,
                use_cuda: bool):
    
    model = QNetwork(env, robust_model, use_cuda)
    if use_cuda:
        model = model.cuda()
    return model
