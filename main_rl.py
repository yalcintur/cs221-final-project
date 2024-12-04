import numpy as np
import torch
from envs.ev2_gym_encoded_env import EV2GymEncoded
from training.train_rl_model import train_rl_model
from training.evaluate import evaluate_rl_model
import joblib
from ev2gym.rl_agent.state import V2G_profit_max_loads
from ev2gym.rl_agent.reward import ProfitMax_TrPenalty_UserIncentives
import gymnasium as gym
from models.autoencoder import Autoencoder 

def main():
    # Step 1: Initialize Environment and Agent
    config_file = "/home/weh4401/EV2Gym/ev2gym-aesac/V2GProfitPlusLoads.yaml"

    input_dim = 112
    latent_dim = 32
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))
    autoencoder.eval()
    scaler = joblib.load("state_scaler.pkl")

    gym.envs.register(id='evs-v1', entry_point='envs.ev2_gym_encoded_env:EV2GymEncoded',
                                kwargs={'config_file': config_file,
                                        'generate_rnd_game': True,
                                        'state_function': V2G_profit_max_loads,
                                        'reward_function': ProfitMax_TrPenalty_UserIncentives,
                                        'scaler': scaler,
                                        'encoder': autoencoder.encoder
                                        })
    env = gym.make('evs-v1')
    
    rl_model = train_rl_model(env, timesteps=1_000_000)

    print("Evaluating trained model...")
    evaluation_results = evaluate_rl_model(env, rl_model, num_steps=112)
    print("Evaluation Results:")
    for key, value in evaluation_results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
