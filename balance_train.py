import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from replay_buffer import ReplayBuffer
from balance_env import GazeboEnv_B

def evaluate(network, epoch, eval_episodes=2):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done= env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward

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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim+action_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim+ action_dim, 800)
        self.layer_5 = nn.Linear(800, 600)
        self.layer_6 = nn.Linear(600, 1)


    def forward(self, s, a):
        sa = torch.cat((s, a), dim=1)

        s1 = F.relu(self.layer_1(sa))
        s1 = F.relu(self.layer_2(s1))
        q1 = F.relu(self.layer_3(s1))

        s2 = F.relu(self.layer_4(sa))
        s2 = F.relu(self.layer_5(s2))
        q2 = F.relu(self.layer_6(s2))

        return q1, q2

# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.iter_count = 0


    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()
                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
            av_loss += loss
        self.iter_count += 1

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


if not os.path.exists("./results_B"):
    os.makedirs("./results_B")
if not os.path.exists("./models_B"):
    os.makedirs("./models_B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
eval_freq = 200  # After how many steps to perform the evaluation
# eval_freq = 5e3
# max_ep = 500  # maximum number of steps per episode
eval_ep = 2  # number of episodes for evaluation
# max_timesteps = 5e6  # Maximum number of steps to perform
max_timesteps = 500
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    500000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # The actor network is updated every n-th step
file_name = "model"  # Path to save and load the model
save_model = True  # Boolean whether or not to save the model
load_model = False  # Boolean whether or not to load the model
buffer_size = 50000  # Size of the replay buffer

# Initialize environment and network
launch_file = "gazebo.launch"  # launch file to start the environment
package = "final"  # package containing the launch file
print("CKPT1")
env = GazeboEnv_B(launch_file,  package)  # 24 is the number of velodyne data, NOT the state dim, state dim = 24 + 4 

time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = 3
action_dim = 2
max_action = 5
print("CKPT2")
# Create the network
network = TD3(state_dim, action_dim, max_action)
print("CKPT3")
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)

# Load model if specified
if load_model:
    try:
        network.load(file_name, "./models_B")
    except:
        print("Could not load the stored model parameters, initializing training with random parameters")
# Evaluation and training variables
evaluations = []
timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1
state = env.reset()
prev_timestep = 0
# Training loop
while timestep < max_timesteps:
    # On termination of episode
    print("TIME_STEP: ",timestep)
    print("----------NOT DONE--------")
    if done:
        prev_timestep = timestep
        print("~~~~~~~~~~DONE~~~~~~~~~~")
        if timestep != 0:
            # Train the network
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        if timesteps_since_eval >= eval_freq:
            # Perform evaluation
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="./models_B")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        # Reset the environment and initialize episode variables
        state = env.reset()
        print("RESETTING ", state.shape)
        # state = network.ensure_fixed_state_size(np.array(state))
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        # Decay exploration noise
        if expl_noise > expl_min:
            expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    # Select action with exploration noise
    # state = network.ensure_fixed_state_size(state) # ensure state size consistency
    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )
    a_in = [action[0], action[1]]

    # Step in the environment
    next_state, reward, done= env.step(a_in)
    reward = reward/(timestep - prev_timestep+1)
    # print("REW ",reward)
    print("FACTOR: ",timestep - prev_timestep+1)

    # Ensure the state size is consistent
    # next_state = network.ensure_fixed_state_size(np.array(next_state))

    # Save the transition in the replay buffer
    replay_buffer.add(state, action, reward, done, next_state)

    # Update the counters
    state = next_state
    episode_reward += reward
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# Final evaluation and model saving
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    print("__SAVING__")
    network.save("%s" % file_name, directory="./models_B")
np.save("./results/%s" % file_name, evaluations)
