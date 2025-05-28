
from myosuite.utils import gym
import torch.nn
import torch.optim
import numpy as np
import time
import random 
import os
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
from torchRL_PPO import make_env, make_ppo_models, make_ppo_models_state
import weights_comp
class RLController:
    def __init__(self, actor_model, device='cpu'):
        self.actor_model = actor_model
        #print(actor_model)
        self.device = device
        self.actor_model.to(self.device)
        self.actor_model.eval()

    def predict(self, state):
        #print(state)
        """if isinstance(state, tuple):
            state = np.array(state[0], dtype=np.float32)
            #print(state)
        state = torch.tensor(state, dtype=torch.float32).to(self.device).squeeze()
        """
        #print(state)
        with torch.no_grad():
            #print(state)
            loc, scale, action, prob = self.actor_model(state)
            #action_sel = (torch.distributions.Normal(loc, scale)).sample().numpy()
            #probs = torch.squeeze(dist.log_prob(action)).item()
            action_sel = action #torch.squeeze(action_sel).item()
            #action = action_probs[2].numpy()# torch.argmax(action_probs[0], dim=-1).item() #action_probs[2].numpy()
            print(loc, scale, action, prob, action_sel)
        return action_sel
    
    def modify_target(self):
        target_qpos = np.array(random.uniform(0, 1.57))
        target_qpos =torch.tensor(target_qpos, dtype=torch.float32).to(self.device).unsqueeze(0)
        return target_qpos
        
def load_actor_model(env_name, model_path, device='cpu'):
    actor, _ = make_ppo_models(env_name)
    actor.load_state_dict(torch.load(model_path))
    actor.to(device)
    
    #actor.load_state_dict(torch.load(model_path, map_location=device))
    return actor

def main():
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    
    # Cargar el modelo del actor entrenado
    actor_model_path = "/Users/achs/Documents/PHD/code/NAIR_Code/code/RL/Pytorch/outputs/2024-09-02/09-55-30/actor_model.pth"
    training_weights = os.path.join(os.path.dirname(actor_model_path), "trained_weights.txt")
    env_name = "ExoLeg40MuscFlexoExtEnv-v0"
    charged_weights = "controller_weights.txt"
    actor_model = load_actor_model(env_name, actor_model_path, device)
    weights_comp.save_weights_to_txt(actor_model, charged_weights)
    weights_comp.compare_files(training_weights, charged_weights)
    
    # Inicializar el controlador con el modelo del actor entrenado
    RL_controller = RLController(actor_model, device=device)
    
    # Crear el entorno
    env = gym.make(env_name, target_qpos=0.5)
    proof_env = make_env(env_name)
    
    state = proof_env.reset()
    env.reset()
    #qpos = [0.005834685602873783, -0.00653851963837641, -1.610314825968498, -0.6692592235178847, 0.00566594464090885, 0.0009114920866788922, 1.4722420075124696, 0.006470171955450531, 0.056191264122378036, -0.1603820765979743, -0.4011910332346481, 0.3400333691081888, -0.16811998188958785, -0.017757503091114304, 0.02614206520113357, -1.3097044945216785]
    #state = env.set_env_state(qpos)
    done = False
    total_reward = 0
    i=0
    actions=[]  
    while not done:
        action = RL_controller.predict(state)
        #print("------",np.array(actions),"------")
        next_state, *_  = proof_env.step(action)
        #print(done)
        state = next_state
        #time.sleep(0.01)
        #env.mj_render()
        actions.append(action)
        if i==2048:
            done = True
            #target_qpos=RL_controller.modify_target()
            #env.set_target_qpos(target_qpos=random.uniform(0, 1.57))
            #env.set_target_qpos(target_qpos=random.uniform(0, 1.57))
            #state = env.random_init()
            i = 0
           
        i = i +1
    #print(f"Recompensa total: {total_reward}")
    for i in actions:
        
        obs, *_ = env.step(np.array(i))
        
        time.sleep(0.01)
        env.mj_render()
        print(obs[7])

    env.close()

if __name__ == "__main__":
    main()