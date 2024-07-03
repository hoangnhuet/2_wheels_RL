import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from balance_env import GazeboEnv_B

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, filename)))

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0  # Random seed number
file_name = "model_actor.pth"  
launch_file = "gazebo.launch"  
package = "final" 
env = GazeboEnv_B(launch_file, package)

time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = 3
action_dim = 2
max_action = 5 # Adjust if necessary for your environment

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./models_B")
except Exception as e:
    raise ValueError("Could not load the stored model parameters: " + str(e))

done = False
episode_timesteps = 0
state = env.reset()

while True:
    action = network.get_action(np.array(state))
    action = action.clip(-max_action, max_action)
    a_in = [action[0] , action[1]]
    next_state, reward, done = env.step(a_in)

    if done:
        state = env.reset()
        done = False
        episode_timesteps = 0
    else:
        state = next_state
        episode_timesteps += 1
