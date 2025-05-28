import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from simulation_interface import SimulationInterface
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
from button_environment import PyBulletAlreadyConnectedException

def convert_nx_to_pyg(G):
    node_attr_dim = 13  # Node feature dimension
    edge_attr_dim = 4   # Edge feature dimension

    # Ensure node features are present
    for node in G.nodes(data=True):
        if 'x' not in node[1]:
            node[1]['x'] = torch.tensor([0] * node_attr_dim, dtype=torch.float)
    
    # Ensure edge features are present
    for u, v, data in G.edges(data=True):
        if 'edge_attr' not in data:
            data['edge_attr'] = torch.tensor([0] * edge_attr_dim, dtype=torch.float)  # Ensure correct dimension

    # Convert to PyTorch Geometric format
    data = from_networkx(G)

    # Extract node features
    if data.x is None or len(G.nodes) == 0:
        #print("No nodes with 'x' attribute found in the graph or graph is empty")
        data.x = torch.empty((0, node_attr_dim), dtype=torch.float)
    else:
        x = torch.stack([G.nodes[node]['x'] for node in G.nodes], dim=0)
        data.x = x
        #print(f"Node features set: {data.x.shape}")

    # Handle the case where there might be no edges
    if len(G.edges) > 0:
        # Extract edge features
        edge_attr = torch.stack([G.edges[edge]['edge_attr'] for edge in G.edges], dim=0)
        data.edge_attr = edge_attr
        #print(f"Edge features set: {data.edge_attr.shape}")
    else:
        data.edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)  # Ensure correct dimension
        #print("No edges in graph, setting empty edge_attr")

    # Ensure edge_index is present
    if data.edge_index is None or data.edge_index.numel() == 0:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        #print("Setting empty edge_index")

    # Ensure batch is present
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        #print("Setting default batch")

    #print(f"convert_nx_to_pyg - x: {data.x.shape}, edge_index: {data.edge_index.shape}, edge_attr: {data.edge_attr.shape}, batch: {data.batch.shape}")

    return data

# Define the neural network for edge attributes
class EdgeTransform(torch.nn.Module):
    def __init__(self, edge_attr_dim, in_channels, out_channels):
        super(EdgeTransform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc1 = torch.nn.Linear(edge_attr_dim, 128)
        self.fc2 = torch.nn.Linear(128, in_channels * out_channels)

    def forward(self, edge_attr):
        edge_attr = F.relu(self.fc1(edge_attr))
        return self.fc2(edge_attr).view(-1, self.in_channels, self.out_channels)  # Ensure correct reshaping

# Define the GNN model(Tool Component Relationship Graph)
class TCRGTransform(torch.nn.Module):
    def __init__(self, node_dim, edge_attr_dim, hidden_dim, output_dim):
        super(TCRGTransform, self).__init__()
        self.edge_net1 = EdgeTransform(edge_attr_dim, node_dim, hidden_dim)
        self.conv1 = NNConv(node_dim, hidden_dim, self.edge_net1, aggr='mean')
        self.pre_pool = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        device = next(self.parameters()).device
        x, edge_index, edge_attr, batch = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device)
        #print(f"Initial x shape: {x.shape}, edge_index shape: {edge_index.shape}, edge_attr shape: {edge_attr.shape}, batch shape: {batch.shape}")

        if x.size(0) == 0:  # Handle empty input
            return torch.zeros((1, self.fc1.out_features), device=device)

        x = x.float()
        edge_attr = edge_attr.float()
        if edge_index.size(1) > 0:  # Check if there are edges
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            #print(f"After conv1, x shape: {x.shape}")

        # Ensure the input to pre_pool has the correct shape
        if x.size(1) != self.pre_pool.in_features:
            x = F.relu(self.conv1(x, edge_index, edge_attr))  # Apply conv1 to get correct shape

        x = F.relu(self.pre_pool(x))  # Transform node features before pooling
        x = global_mean_pool(x, batch)  # Pooling
        #print(f"After pooling, x shape: {x.shape}")
        x = F.relu(self.fc1(x))
        return x

class EnvironmentStateTransform(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnvironmentStateTransform, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    

class PPOMemory():
    """
    Memory for PPO
    """
    def  __init__(self, batch_size):
        self.states = []
        self.actions= []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []
        
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size) 
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states),np.array(self.actions),\
               np.array(self.action_probs),np.array(self.vals),np.array(self.rewards),\
               np.array(self.dones),batches
    
    def store_memory(self,state,action,action_prob,val,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.vals.append(val)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions= []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []

class ActorNwk(nn.Module):
    def __init__(self,input_dim,out_dim,
                 adam_lr,
                 chekpoint_file,
                 hidden1_dim=256,
                 hidden2_dim=256
                 ):
        super(ActorNwk, self).__init__()

        self.actor_nwk = nn.Sequential(
            nn.Linear(input_dim,hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim,hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim,out_dim),  
            nn.Softmax(dim=-1)
        )

        self.checkpoint_file = chekpoint_file
        self.optimizer = torch.optim.Adam(params=self.actor_nwk.parameters(),lr=adam_lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    
    def forward(self,state):
        out = self.actor_nwk(state)
        dist = Categorical(out)
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNwk(nn.Module):
    def __init__(self,input_dim,
                 adam_lr,
                 chekpoint_file,
                 hidden1_dim=256,
                 hidden2_dim=256
                 ):
        super(CriticNwk, self).__init__()

        self.critic_nwk = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, 1),
        )

        self.checkpoint_file = chekpoint_file
        self.optimizer = torch.optim.Adam(params=self.critic_nwk.parameters(),lr=adam_lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    
    def forward(self,state):
        out = self.critic_nwk(state)
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent():
    def __init__(
        self, 
        gamma, 
        policy_clip,
        lamda, 
        adam_lr,
        n_epochs, 
        batch_size, 
        tcrg_node_attr_dim,
        tcrg_edge_attr_dim,
        tcrt_hidden_dim = 256,
        tcrt_output_dim = 256,
        environmet_state_dim = 8,
        environmet_state_hidden_dim = 8,
        environmet_state_output_dim = 4,
        action_dim = 4
    ):
        self.gamma = gamma 
        self.policy_clip = policy_clip
        self.lamda  = lamda
        self.n_epochs = n_epochs

        # Total input dimension for actor and critic
        combined_input_dim = tcrt_output_dim + environmet_state_output_dim

        self.actor = ActorNwk(
            input_dim=combined_input_dim,
            out_dim=action_dim,
            adam_lr=adam_lr,
            chekpoint_file='/home/abiantorres/Documents/GitHub/rl-for-tool-generation/models/robot/rl_model/actor'
        )
        self.critic = CriticNwk(
            input_dim=combined_input_dim,
            adam_lr=adam_lr,
            chekpoint_file='/home/abiantorres/Documents/GitHub/rl-for-tool-generation/models/robot/rl_model/critic'
        )
        self.memory = PPOMemory(batch_size)

        self.tcrt = TCRGTransform(
            node_dim=tcrg_node_attr_dim, 
            edge_attr_dim=tcrg_edge_attr_dim, 
            hidden_dim=tcrt_hidden_dim, 
            output_dim=tcrt_output_dim
        )

        self.env_state_transform = EnvironmentStateTransform(
            input_dim=environmet_state_dim,
            hidden_dim=environmet_state_hidden_dim,
            output_dim=environmet_state_output_dim
        )

    def store_data(self, state, action, action_prob, val, reward, done):
        self.memory.store_memory(state, action, action_prob, val, reward, done)

    def save_models(self):
        print('... Saving Models ......')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print('... Loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    
    
    def get_state_transform(self, states):
        device = self.actor.device  # Ensuring we use the same device

        # Process TCRG graphs
        tcrg_graphs = [convert_nx_to_pyg(state['tool_graph']).to(device) for state in states]
        tcrg_vectors = [self.tcrt(graph) for graph in tcrg_graphs]

        # Handle the case where tcrg_vectors is empty
        if len(tcrg_vectors) == 0:
            tcrg_vectors = torch.zeros((len(states), self.tcrt.fc1.out_features), device=device)
        else:
            tcrg_vectors = torch.stack(tcrg_vectors).to(device)

        # Process environment button features
        env_button_features = torch.tensor(
            [state['environment_vector']['button_position'] + state['environment_vector']['button_orientation'] for state in states],
            dtype=torch.float32
        ).to(device)  # Move env_button_features to the same device

        env_state_vectors = self.env_state_transform(env_button_features).to(device)

        # Ensure dimensions match for concatenation
        if len(tcrg_vectors.size()) == 2:
            tcrg_vectors = tcrg_vectors.unsqueeze(1)
        if len(env_state_vectors.size()) == 2:
            env_state_vectors = env_state_vectors.unsqueeze(1)

        # Expand env_state_vectors to match dimensions of tcrg_vectors
        env_state_vectors = env_state_vectors.expand(-1, tcrg_vectors.size(1), -1)

        combined_state = torch.cat((tcrg_vectors, env_state_vectors), dim=-1)
        return combined_state.to(device)

    def choose_action(self, state):
        combined_state = self.get_state_transform([state])

        dist = self.actor(combined_state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()

        value = self.critic(combined_state)
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def calculate_advantage(self, reward_arr, value_arr, dones_arr):
        time_steps = len(reward_arr)
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(0, time_steps-1):
            discount = 1
            running_advantage = 0
            for k in range(t, time_steps-1):
                if int(dones_arr[k]) == 1:
                    running_advantage += reward_arr[k] - value_arr[k]
                else:
                    running_advantage += reward_arr[k] + (self.gamma * value_arr[k+1]) - value_arr[k]

                running_advantage = discount * running_advantage
                discount *= self.gamma * self.lamda
            
            advantage[t] = running_advantage
        advantage = torch.tensor(advantage).to(self.actor.device)
        return advantage
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, value_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            advantage_arr = self.calculate_advantage(reward_arr, value_arr, dones_arr)
            values = torch.tensor(value_arr).to(self.actor.device)
            for batch in batches:
                batch_states = [state_arr[i] for i in batch]
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                combined_states = self.get_state_transform(batch_states)

                dist = self.actor(combined_states)
                critic_value = self.critic(combined_states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage_arr[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage_arr[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage_arr[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory() 

env = SimulationInterface(pybullet_gui = False, truncated_step = 5)
agent = Agent(
    action_dim=len(env.actions_dict), 
    batch_size=32,
    n_epochs=10,
    policy_clip=0.2,
    gamma=0.99,
    lamda=0.95, 
    adam_lr=0.0003,
    tcrg_node_attr_dim=13,
    tcrg_edge_attr_dim=4,
    tcrt_hidden_dim=32,
    tcrt_output_dim=16,
    environmet_state_dim=7,
    environmet_state_hidden_dim=8,
    environmet_state_output_dim=4
)
agent.load_models()
env.start_pybullet_simulation(gui=False)

N_EPISODES = 1000000
n_steps    = 0
score_history = []
learn_iters = 0
best_score = -np.inf
N = 100


"""plt.ion()  # Interactive mode on
fig, ax = plt.subplots(3, 1, figsize=(10, 8))"""

# Initialize empty data arrays
"""reward_data = []
step_data = []
episode_data = []
"""
"""# Accumulated reward plot
ax[0].set_title('Accumulated Reward')
reward_line, = ax[0].plot([], [], 'b-')
ax[0].set_xlim(0, 100)
ax[0].set_ylim(0, 1)
ax[0].set_xlabel('Step')
ax[0].set_ylabel('Reward')

# Step plot
ax[1].set_title('Step')
step_line, = ax[1].plot([], [], 'r-')
ax[1].set_xlim(0, 100)
ax[1].set_ylim(0, 1)
ax[1].set_xlabel('Step')
ax[1].set_ylabel('Steps')

# Episode plot
ax[2].set_title('Episode')
episode_line, = ax[2].plot([], [], 'g-')
ax[2].set_xlim(0, 100)
ax[2].set_ylim(0, 1)
ax[2].set_xlabel('Step')
ax[2].set_ylabel('Episode')

plt.tight_layout()
"""

"""def update_plot(step, episode, accumulated_reward):
    reward_data.append(accumulated_reward)
    step_data.append(step)
    episode_data.append(episode)

    reward_line.set_xdata(range(len(reward_data)))
    reward_line.set_ydata(reward_data)

    step_line.set_xdata(range(len(step_data)))
    step_line.set_ydata(step_data)

    episode_line.set_xdata(range(len(episode_data)))
    episode_line.set_ydata(episode_data)

    for a in ax:
        a.relim()
        a.autoscale_view()

    plt.draw()
    plt.pause(0.01)
"""

for i in range(N_EPISODES):
    try:
        env.start_pybullet_simulation(gui=False)
    except PyBulletAlreadyConnectedException as e:
        pass
    env.reset()
    current_state = env.get_state()
    terminated, truncated = False, False
    done = False
    score = 0
    steps = 0
    while not done:
        action, prob, val = agent.choose_action(current_state)
        next_state, terminated, reward, truncated, exception = env.step(action)
        done = True if (terminated or truncated) else False
        n_steps += 1
        score += reward
        steps += 1
        agent.store_data(current_state, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        current_state = next_state

        # Update the plot for each iteration
        print("steps", steps, "i", i, "score", score, "action", action, "done", done, "truncated", truncated, "exception", exception)

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()
    
    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
          'time_steps', n_steps, 'learning_steps', learn_iters)