
from myosuite.utils import gym
import numpy as np
import time
from torchRL_PPO import make_env, make_ppo_models
import os
import torch
import weights_comp

def load_actor_model(env_name, model_path, device='cpu'):
    actor, _ = make_ppo_models(env_name)
    actor.load_state_dict(torch.load(model_path, map_location=device))
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
    # Crear el entorno
    env = make_env(env_name)
    # Define policy architecture

    #actor_model.eval()
    tensor = env.rollout(1000, actor_model)   
    actions = tensor["action"].detach().numpy().squeeze()
    env = gym.make(env_name)
    env.reset()
    #print(actions)
    for i in actions:
        
        env.step(np.array(i))
        
        time.sleep(0.01)
        env.mj_render('human')
    #print(f"Recompensa total: {total_reward}")
    env.close()
    
if __name__ == "__main__":
    main()