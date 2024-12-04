import numpy as np
import torch
import joblib
import gymnasium as gym
from tqdm import tqdm
from envs.ev2_gym_encoded_env import EV2GymEncoded
from models.heuristic_agent import ChargeAsFastAsPossible
from training.data_collection import collect_data, preprocess_data, load_and_preprocess_data
from training.train_autoencoder import train_autoencoder
from training.train_rl_model import train_rl_model
from training.evaluate import evaluate_rl_model
from ev2gym.rl_agent.reward import ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.state import V2G_profit_max_loads
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG = {
    "config_file": "/home/weh4401/EV2Gym/ev2gym-aesac/V2GProfitPlusLoads.yaml",
    "episodes": 1000,
    "latent_dim": 32,
    "autoencoder_epochs": 1000,
    "autoencoder_batch_size": 8192,
    "autoencoder_patience": 10,
    "rl_timesteps": 5000,
    "evaluation_steps": 112,
    "state_scaler_path": "state_scaler.pkl",
    "autoencoder_path": "autoencoder.pth",
    "gym_env_id": "evs-v1",
    "num_workers": 4,
}

def env_creator():
    return gym.make(CONFIG["gym_env_id"])



def main():
    logging.info("Initializing environment and agent...")
    config_file = CONFIG["config_file"]
    env_org = EV2GymEncoded(config_file=config_file)

    gym.envs.register(
        id=CONFIG["gym_env_id"],
        entry_point="envs.ev2_gym_encoded_env:EV2GymEncoded",
        kwargs={
            "config_file": config_file,
            "generate_rnd_game": True,
            "state_function": V2G_profit_max_loads,
            "reward_function": ProfitMax_TrPenalty_UserIncentives,
        },
    )
    env = gym.make('evs-v1')

    agent = ChargeAsFastAsPossible()

    logging.info("Collecting data from environment...")
    #state_data = collect_data(env_org=env_org, env=env, agent=agent, episodes=CONFIG["episodes"], num_workers=CONFIG["num_workers"])

    logging.info("Preprocessing collected data...")
    #train_data, test_data = preprocess_data(state_data)
    train_data, test_data, scaler = load_and_preprocess_data()
    
    logging.info("Training autoencoder...")
    input_dim = train_data.shape[1]
    autoencoder = train_autoencoder(
        train_data=train_data,
        val_data=test_data,
        input_dim=input_dim,
        latent_dim=CONFIG["latent_dim"],
        epochs=CONFIG["autoencoder_epochs"],
        batch_size=CONFIG["autoencoder_batch_size"],
        patience=CONFIG["autoencoder_patience"],
    )

    logging.info("Saving scaler and encoder...")
    torch.save(autoencoder.state_dict(), CONFIG["autoencoder_path"])

    def evaluate_reconstruction(data):
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32)
            _, reconstructed = autoencoder(data_tensor)
            mse = ((reconstructed.numpy() - data) ** 2).mean()
        return mse
    
    train_mse = evaluate_reconstruction(train_data)
    test_mse = evaluate_reconstruction(test_data)

    print(f"Train Reconstruction MSE: {train_mse}")
    print(f"Test Reconstruction MSE: {test_mse}")


if __name__ == "__main__":
    main()
