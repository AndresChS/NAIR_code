"""	
This model is generated with tacking the Myosuite conversion of [Rajagopal's full body gait model](https://github.com/opensim-org/opensim-models/tree/master/Models/RajagopalModel) as close
reference.
	Model	  :: MyoLeg 1 Dof Exo (MuJoCoV2.0)
	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
	source	:: https://github.com/AndresChS/NAIR_Code
"""

import hydra

import torch.nn
import torch.optim
import numpy as np

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator


from torch.distributions.categorical import Categorical
#from torch_geometric.utils import from_networkx
#import matplotlib.pyplot as plt
#from button_environment import PyBulletAlreadyConnectedException
# ====================================================================
# Environment utils
# --------------------------------------------------------------------

from myosuite.utils import gym

def make_env(env_name="", device="cpu"):
    env = GymWrapper(gym.make("ExoLeg40MuscFlexoExtEnv-v0"), device=device)
    env = TransformedEnv(env)
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat())
    """
    print("observation_spec:", env.observation_spec)
    print("reward_spec:", env.reward_spec)
    print("input_spec:", env.input_spec)
    print("action_spec (as defined by input_spec):", env.action_spec)
    rollout = env.rollout(3)
    print("rollout of three steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)
    """
    return env

# ====================================================================
# Model utils
# --------------------------------------------------------------------

@hydra.main(config_path=".", config_name="config_PPO")
#def main(cfg: "DictConfig"):  # noqa: F821

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

class ActorNwk(torch.nn.Module):
    def __init__(self,input_dim,out_dim,
                 adam_lr,
                 chekpoint_file,
                 hidden1_dim=256,
                 hidden2_dim=256
                 ):
        super(ActorNwk, self).__init__()

        self.actor_nwk = torch.nn.Sequential(
            torch.nn.Linear(input_dim,hidden1_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden1_dim,hidden2_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden2_dim,out_dim),  
            torch.nn.Softmax(dim=-1)
        )

        self.checkpoint_file = chekpoint_file
        self.optimizer = torch.optim.Adam(params=self.actor_nwk.parameters(),lr=adam_lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    
    def forward(self,state):
        loc = torch.tanh(self.actor_nwk(state))  
        scale = torch.exp(self.actor_nwk(state))   
        return loc, scale

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNwk(torch.nn.Module):
    def __init__(self,input_dim,
                 adam_lr,
                 chekpoint_file,
                 hidden1_dim=256,
                 hidden2_dim=256
                 ):
        super(CriticNwk, self).__init__()

        self.critic_nwk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden1_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden1_dim, hidden2_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden2_dim, 1),
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

# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------
class Agent():
    def __init__(
        self, 
        gamma, 
        policy_clip,
        lamda, 
        adam_lr,
        n_epochs, 
        batch_size, 
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
        combined_input_dim = environmet_state_output_dim

        self.actor = ActorNwk(
            input_dim=combined_input_dim,
            out_dim=action_dim,
            adam_lr=adam_lr,
            chekpoint_file='/Users/achs/Documents/PHD/code/NAIR_Code/code/RL/Pytorch/RL_model/actor'
        )
        self.critic = CriticNwk(
            input_dim=combined_input_dim,
            adam_lr=adam_lr,
            chekpoint_file='/Users/achs/Documents/PHD/code/NAIR_Code/code/RL/Pytorch/RL_model/critic'
        )
        self.memory = PPOMemory(batch_size)

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

    def choose_action(self, state):
        # Actor Network part / action & probs
        tensor_state = torch.tensor(state, dtype=torch.float32) #self.get_state_transform([state])
        loc, scale = self.actor(tensor_state)
        dist = torch.distributions.Normal(loc, scale)
        #print("-------",dist,"-----")
        action = dist.sample()
        #print("-------",action,"-----")
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        #print("-------",probs,"-----")
        # Critic Network part / value
        value = self.critic(tensor_state)
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
                #print(actions)
                combined_states = torch.tensor(np.array(batch_states)).float() #self.get_state_transform(batch_states)
                
                loc, scale = self.actor(combined_states)
                normal_dist = torch.distributions.Normal(loc, scale)
                critic_value = self.critic(combined_states)
                critic_value = torch.squeeze(critic_value)

                new_probs = normal_dist.log_prob(actions)
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


@hydra.main(config_path=".", config_name="config_PPO")
def main(cfg):  # noqa: F821
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    env = make_env(cfg.env.env_name, device)
    myosuite_env = gym.make(cfg.env.env_name)
    print("num_outputs----", env.observation_spec["observation"].shape[-1])
    print("num_inputs----", env.action_spec.shape[-1])
    agent = Agent(
        action_dim=env.action_spec.shape[-1], 
        batch_size=32,
        n_epochs=10,
        policy_clip=0.2,
        gamma=0.99,
        lamda=0.95, 
        adam_lr=cfg.optim.lr,
        environmet_state_dim=7,
        environmet_state_hidden_dim=8,
        environmet_state_output_dim=env.observation_spec["observation"].shape[-1]
    )
    
    

    N_EPISODES = 100000
    n_steps    = 0
    score_history = []
    learn_iters = 0
    best_score = -np.inf
    N = 100

# ====================================================================
# Train Loop
# --------------------------------------------------------------------
    for i in range(cfg.collector.total_frames):
        
        env.reset()
        myosuite_env.reset()
        current_state = myosuite_env.get_obs()
        current_state_torch = env.get_obs()
        #print(current_state)
        terminated, truncated = False, False
        done = False
        score = 0
        steps = 0
        while not done:
            action, prob, val = agent.choose_action(current_state)
            #print(action)
            next_state, reward, terminal, truncated, info = myosuite_env.step(np.array(action))
            #print("hola")
            done = True if (terminated or truncated) else False
            n_steps += 1
            score += reward
            steps += 1
            agent.store_data(current_state, action, prob, val, reward, done)
            if n_steps % N == 0:
                #print("hola")
                agent.learn()
                learn_iters += 1
            current_state = next_state

            # Update the plot for each iteration
            #print("steps", steps, "i", i, "score", score, "action", action, "done", done, "truncated", truncated, "exception")

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)
        
if __name__ == "__main__":
    main()
